"""MNIST difficulty 5 (band 1.90%): a kernel-ridge teacher distilled into a semi-supervised Ladder network.

Like the difficulty-4 entry, but 800 steps, teacher weight 0.25, AGOP^0.75 student inputs, plain autograd and
cheap-compile settings (a cold A100 compile fits the warm-up limit). Inputs are whitened+rotated: it learns a metric.
 1. Kernel teacher (after @islamborghini's kernel-pcg): rows are partly normalised; RBF kernel ridge is fitted by
    Nystrom-preconditioned CG, and the AGOP M of the fit (mean outer product of its input gradients) sets the
    metric M^0.75 for the next fit, twice. A fit on all train rows scores the test rows; confident ones (top-2
    margin >= 0.15) join as pseudo-labels in a refit that scores every test row. Few CG steps: stopping early
    regularises.
 2. Student inputs: the teacher's metric coordinates rownorm(x) @ L, rescaled.
 3. Ladder network: a 60-1024-512-256-10 encoder with batch norm and Gaussian noise on every layer; a decoder of
    per-unit MLP combinators denoises down to the input, so unlabelled rows shape the features. Loss: cross-entropy
    on labelled rows (sampled partly by recent loss) plus, on train + test rows, an input-level reconstruction cost
    and a cross-entropy toward the true labels (train) or the teacher's softened scores (test). 800 Adam steps.
 4. Predict: BN statistics from training rows, a clean (noise-free) encoder pass, argmax.
test_x is used transductively, without labels (pseudo-labels, distillation targets, unlabelled Ladder rows).

Speed: fp16 tensor-core kernel panels; the teacher and the train step each replay one CUDA graph; the train step is
torch.compile'd with bf16 hidden activations. Seeds are fixed, but training is nondeterministic (order-dependent
GPU ops): repeat calls on the same data flip tens of test labels, at about the same error.

Credits: Ladder recipe from @SethTS's Ladder records and Rasmus et al. 2015 ("Semi-Supervised Learning with Ladder
Networks").
"""
import math
import os

import torch
import torch.nn.functional as F

# fp32 GEMMs stay true fp32 (no TF32); fp16 GEMMs accumulate in fp32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# RBF width, ridge, Nystrom rank, AGOP rows, identity blend, row-norm power
GAMMA, RIDGE, RANK, SAMPLES, BLEND, NPOW = 0.02, 0.1, 256, 512, 0.02, 0.6
AGOP_N, M_STEPS, BSTEPS, RSTEPS = (3000, 10000), (16, 16), 14, 16  # PCG steps: metric fits, base fit, refit
# pseudo-label margin, panel rows, Newton-Schulz steps, exponent shift
PSEUDO_CONF, P, NS_ITERS, SHIFT = 0.15, 2000, 15, 3.0
F32 = torch.float32
M, DIMS = 1, (60, 1024, 512, 256, 10)
STEPS, BATCH, LR, BETA1, DECAY_START = 800, 2000, 0.011, 0.8, 2 / 3
NOISE, SCALE, RECON, EPS, CSTD, SEED = 0.25, 0.6, 2000.0, 1e-10, 0.025, 11
MIX, MIX_A, MIX_B, MIX_C = 0.5, 0.1, 0.6, 0.9  # loss-weighted sampling share, schedule
TW, TEACH_TAU = 0.25, 0.1  # distillation weight, temperature
COMPILE, ACT = True, torch.bfloat16
# train-step inductor options: combo kernels + cheap cold compile
IND = {"combo_kernels": True, "triton.autotune_pointwise": False, "pattern_matcher": False, "fx_graph_cache": False,
       "autotune_local_cache": False, "reorder_for_peak_memory": False, "size_asserts": False, "scalar_asserts": False,
       "alignment_asserts": False}
CACHE = {}  # CUDA graphs, keyed by shape


# ---- kernel teacher

def chol(a):
    return torch.linalg.cholesky_ex(a)[0]  # no error check, no host sync: graph-safe


def rownorm(x):
    return x * (x.shape[1] ** 0.5 / x.norm(dim=1, keepdim=True).clamp_min(1e-6)) ** NPOW


def aug(a, b, shift=0.0):
    """A @ B = shift - GAMMA |a_i - b_j|^2"""
    sa = a.square().sum(1, keepdim=True)
    sb = b.square().sum(1, keepdim=True)
    A = torch.cat([a, sa, torch.ones_like(sa)], 1)
    B = torch.cat([(2 * GAMMA) * b, torch.full_like(sb, -GAMMA), shift - GAMMA * sb], 1).T.contiguous()
    return A, B


def kern(a, b):
    A, B = aug(a, b)
    return (A @ B).exp_()


def h3_ops(z):
    """fp16 hi/lo split: Ah Bh + Ah Bl + Al Bh gives the exponent to ~fp32 accuracy in one fp16 GEMM"""
    A, B = aug(z, z, SHIFT)
    Ah, Bh = A.half(), B.half()
    Al, Bl = (A - Ah.float()).half(), (B - Bh.float()).half()
    return torch.cat([Ah, Ah, Al], 1), torch.cat([Bh, Bl, Bh], 0)


def panels(A3, B3, pp, rows, diag):
    """upper row panels (rows i:i+pp, columns i:) of the fp16 kernel; diag = kernel + ridge"""
    out = []
    for i in rows:
        r = min(pp, A3.shape[0] - i)
        pi = torch.mm(A3[i:i + r], B3[:, i:])
        pi.exp_()
        pi[:, :r].diagonal().fill_(diag)
        out.append(pi)
    return out


class Sym:
    """K[:n, :n] from upper row panels of pp rows (n a multiple of pp, or the full size)"""

    def __init__(self, pans, pp, n):
        self.pans, self.pp, self.n = pans, pp, n

    def mv(self, dh):
        pp, n = self.pp, self.n
        y = torch.empty(n, dh.shape[1], device=dh.device)
        p0 = self.pans[0]
        r0 = min(p0.shape[0], n)
        torch.mm(p0[:r0, :n], dh, out_dtype=F32, out=y[:r0])
        if r0 < n:
            torch.mm(p0[:, r0:n].T, dh[:r0], out_dtype=F32, out=y[r0:n])
        for j in range(1, len(self.pans)):
            a = j * pp
            if a >= n:
                break
            pj = self.pans[j]
            e = min(a + pj.shape[0], n)
            torch.addmm(y[a:e], pj[:, :n - a], dh[a:n], out_dtype=F32, out=y[a:e])
            if e < n:
                torch.addmm(y[e:n], pj[:, e - a:n - a].T, dh[a:e], out_dtype=F32, out=y[e:n])
        return y

    def cross(self, w, lo, hi):
        """K[lo:hi, :lo] w from the panels (coefficients rescaled into fp16 range)"""
        s = w.abs().amax(0, keepdim=True).clamp_min(1e-30)
        wh = (w / s).half()
        y = torch.empty(hi - lo, w.shape[1], device=w.device)
        for j, pj in enumerate(self.pans):
            a = j * self.pp
            if a >= lo:
                break
            if j == 0:
                torch.mm(pj[:, lo - a:hi - a].T, wh[a:a + pj.shape[0]], out_dtype=F32, out=y)
            else:
                torch.addmm(y, pj[:, lo - a:hi - a].T, wh[a:a + pj.shape[0]], out_dtype=F32, out=y)
        return y * s


def upd_comp(ap, d, w, r, nrm, v, vt, mask):
    """one PCG iteration given ap = op.mv(d)"""
    if mask is not None:
        ap = ap * mask[:, None]
    a = nrm / (d * ap).sum(0)
    w = w + d * a
    r = r - ap * a
    p = r - v @ (vt @ r)
    nxt = (r * p).sum(0)
    d = p + d * (nxt / nrm)
    return w, r, d, nxt, d.half()


def lfac(kmm, jit):
    """L^-T, L L^T = kmm + jit I: u = Km L^-T has u u^T ~ K (Nystrom)"""
    lm = kmm.clone()
    lm.diagonal().add_(jit)
    eye = torch.eye(len(kmm), device=kmm.device)
    return torch.linalg.solve_triangular(chol(lm), eye, upper=False).T


def pcg(op, t, lam, steps, u, mask=None):
    """Nystrom-PCG for (D K D + lam I) w = D t; op.mv(dh) = (K + lam I) d; u = Km L^-T"""
    if mask is not None:
        u = u * mask[:, None]
        t = t * mask[:, None]
    small = u.T @ u
    small.diagonal().add_(lam)
    eye = torch.eye(len(small), device=u.device)
    v = u @ torch.linalg.solve_triangular(chol(small), eye, upper=False).T
    vt = v.T.contiguous()  # r - v vt r = lam (u u^T + lam I)^-1 r
    w = torch.zeros_like(t)
    r = t.clone()
    d = torch.addmm(r, v, vt @ r, alpha=-1)
    nrm = torch.linalg.vecdot(r, d, dim=0)
    dh = d.half()
    for _ in range(steps):
        w, r, d, nrm, dh = upd_comp(op.mv(dh), d, w, r, nrm, v, vt, mask)
    return w


def sqrtm(a):
    """coupled Newton-Schulz square root, run as the sign iteration on [[0, A/c], [I, 0]]"""
    c = a.norm()
    n = len(a)
    x = torch.zeros(2 * n, 2 * n, device=a.device)
    x[:n, n:] = a / c
    x[n:, :n] = torch.eye(n, device=a.device)
    i15 = 1.5 * torch.eye(2 * n, device=a.device)
    for _ in range(NS_ITERS):
        x = x @ torch.addmm(i15, x, x, alpha=-0.5)
    return x[:n, n:] * c.sqrt()


def agop(transform, z, w, fac):
    """factor L (L L^T = blend(M^0.75)) of the AGOP of f = sum_j w_j k(., z_j) at the first SAMPLES rows"""
    s = fac.shape[0]
    wp = torch.cat([w, (w[:, :, None] * z[:, None, :]).reshape(len(z), -1)], 1)
    r = fac.float() @ wp
    c = w.shape[1]
    # grad f(x) ~ sum_j w_j k(x, z_j) (z_j - x), in raw coordinates
    g = r[:, c:].reshape(s, c, -1) - r[:, :c, None] * z[:s, None, :]
    dim = transform.shape[0]
    g = (g @ transform.T).reshape(-1, dim)
    m = g.T @ g
    sq = sqrtm(m)
    m = sq @ sqrtm(sq)
    m = (m + m.T) / 2
    m = m * (dim / m.trace().clamp_min(1e-12))
    m = (1 - BLEND) * m + BLEND * torch.eye(dim, device=z.device)
    return chol(m)


def sym_scores(train_x, train_y, test_x):
    c = math.exp(SHIFT)  # panels hold c k: lam, jit, diag scale by c
    lam, jit, diag = RIDGE * c, 1e-4 * c, c * (1.0 + RIDGE)
    with torch.no_grad():
        x, q = rownorm(train_x.float()), rownorm(test_x.float())
        n = len(x)
        y = F.one_hot(train_y.long(), 10).float()
        transform = torch.eye(x.shape[1], device=x.device)
        for m, st in zip(AGOP_N, M_STEPS):
            z = x[:m] @ transform
            km = kern(z, z[:RANK]) * c
            pm = P if m % P == 0 else m
            A3, B3 = h3_ops(z)
            pan = panels(A3, B3, pm, range(0, m, pm), diag)
            w = pcg(Sym(pan, pm, m), y[:m], lam, st, km @ lfac(km[:RANK], jit))
            fac = pan[0][:SAMPLES].clone()
            fac.diagonal().fill_(c)
            del pan
            transform = agop(transform, z, w, fac)
        za = torch.cat([x, q]) @ transform
        N = len(za)
        A3, B3 = h3_ops(za)
        pan = panels(A3, B3, P, range(0, n, P), diag)
        km = kern(za, za[:RANK]) * c
        lf = lfac(km[:RANK], jit)
        # test panels are only for the refit: build them on a side stream
        main = torch.cuda.current_stream()
        ss = torch.cuda.Stream()
        ss.wait_stream(main)
        with torch.cuda.stream(ss):
            pan_t = panels(A3, B3, P, range(n, N, P), diag)
            ua = km @ lf
        opx = Sym(pan, P, n)
        w = pcg(opx, y, lam, BSTEPS, km[:n] @ lf)
        s = opx.cross(w, n, N)
        top = s.topk(2, 1).values
        keep = (top[:, 0] - top[:, 1]) >= PSEUDO_CONF
        mask = torch.cat([torch.ones(n, device=x.device), keep.float()])
        t = F.one_hot(s.argmax(1), 10).float()
        main.wait_stream(ss)
        opn = Sym(pan + pan_t, P, N)
        a = pcg(opn, torch.cat([y, t]), lam, RSTEPS, ua, mask)
        sc = a.abs().amax(0, keepdim=True).clamp_min(1e-30)
        ah = (a / sc).half()
        out = opn.mv(ah)[n:]  # (cK + lam I) a on the test rows
        s_all = (out - lam * ah[n:].float()) * sc  # = cK[n:, :] a = the refit prediction (a carries 1/c)
        return s_all, transform


def teacher(train_x, train_y, test_x):
    """sym_scores as one CUDA graph per shape"""
    key = ("teacher", tuple(train_x.shape), tuple(test_x.shape), str(train_x.device))
    if key not in CACHE:
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                sym_scores(*static)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = sym_scores(*static)
        CACHE[key] = (g, static, out)
    g, static, out = CACHE[key]
    static[0].copy_(train_x)
    static[1].copy_(train_y)
    static[2].copy_(test_x)
    g.replay()
    return out[0].clone(), out[1].clone()


# ---- ladder networks

def bn(u):
    var, mean = torch.var_mean(u, dim=-2, unbiased=False, keepdim=True)
    return (u - mean) * torch.rsqrt(var + EPS)


def lrelu(t):
    return F.leaky_relu(t, 0.1)


def combine(u, lat, p):
    """Per-unit MLP combinator g(u, z~) of the decoder; u, lat (M, B, s); p (17, M, 1, s)."""
    v = bn(u)
    x2 = v * lat
    a0 = lrelu(v * p[0] + lat * p[2] + x2 * p[4] + p[6])
    a1 = lrelu(v * p[1] + lat * p[3] + x2 * p[5] + p[7])
    c0 = lrelu(a0 * p[8] + a1 * p[10] + p[12])
    c1 = lrelu(a0 * p[9] + a1 * p[11] + p[13])
    return c0 * p[14] + c1 * p[15] + p[16]


def mm(h, w, act=None):
    """h (M, ..., a) @ w (M, a, b) as one bmm: in autocast with output dtype act, or plain fp32 if act is None."""
    s = h.shape
    with torch.autocast("cuda", dtype=act or torch.float16, enabled=act is not None):
        out = torch.bmm(h.reshape(s[0], -1, s[-1]), w)
    return (out if act is not None else out.float()).reshape(*s[:-1], w.shape[-1])


class Ladder:
    def __init__(self, n, nq, device):
        self.n, self.nq, self.device = n, nq, device
        zeros = lambda *s: torch.zeros(*s, device=device)
        self.enc = [zeros(M, a, b) for a, b in zip(DIMS[:-1], DIMS[1:])]
        self.dec = [zeros(M, b, a) for a, b in zip(DIMS[:-1], DIMS[1:])]
        self.beta = [zeros(M, 1, b) for b in DIMS[1:]]
        self.gamma = zeros(M, 1, DIMS[-1])
        self.comb = [zeros(17, M, 1, s) for s in DIMS]
        self.params = self.enc + self.dec + self.beta + [self.gamma] + self.comb
        for p in self.params:
            p.requires_grad_(True)
        self.x = zeros(n, DIMS[0])
        self.y = zeros(n, 10)
        self.pool = zeros(n + nq, DIMS[0])
        self.ptar = zeros(n + nq, 10)
        self.scores = zeros(M, n)
        self.offs = (torch.arange(M, device=device) * n)[:, None]
        self.counter = torch.zeros((), dtype=torch.long, device=device)
        self.iu = torch.zeros(STEPS, M, BATCH, dtype=torch.long, device=device)
        frac = torch.arange(STEPS, device=device, dtype=torch.float32) / STEPS
        self.lrs = LR * ((1 - frac) / (1 - DECAY_START)).clamp(max=1.0)  # flat, then linear decay
        ramp = ((MIX_C - frac) / (MIX_C - MIX_B)).clamp(0, 1)
        self.mixs = MIX * torch.where(frac < MIX_A, 0.0, torch.where(frac < MIX_B, 1.0, ramp))
        self.lr = torch.tensor(LR, device=device)  # read by capturable Adam
        self.opt = torch.optim.Adam(self.params, lr=self.lr, betas=(BETA1, 0.999), fused=True, capturable=True)
        self.lossfn = torch.compile(self.loss, dynamic=False, options=IND) if COMPILE else self.loss
        self.reset()
        # warm-up (compiles), then capture one step
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.opt.zero_grad(set_to_none=True)
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.step()

    def loss(self, xl, yl, xu, tu):
        """bf16 (ACT) hidden activations and bmms; the 60-d input level and the losses in fp32"""
        beta, gamma = [b.to(ACT) for b in self.beta], self.gamma.to(ACT)
        comb = [p.to(ACT) for p in self.comb]
        h = torch.stack((xl, xu), 1)  # labelled, unlabelled (BN per half)
        h = h + NOISE * torch.randn_like(h)
        lat = [h[:, 1]]
        h = h.to(ACT)
        top = len(self.enc) - 1
        for i, w in enumerate(self.enc):
            z = bn(mm(h, w, ACT))
            z = z + NOISE * torch.randn_like(z)
            lat.append(z[:, 1])
            b = beta[i][:, None]
            h = (z + b) * gamma[:, None] if i == top else F.relu(z + b)
        hf = h.float()
        ce = -(yl * F.log_softmax(hf[:, 0], -1)).sum(-1)
        teach = -(tu * F.log_softmax(hf[:, 1], -1)).sum(-1).mean(-1).sum()
        # decoder on the unlabelled half; only the input level is scored
        r = combine(F.softmax(hf[:, 1], -1).to(ACT), lat[-1], comb[-1])
        for i in range(top, 0, -1):
            r = combine(mm(r, self.dec[i], ACT), lat[i], comb[i])
        r = combine(mm(r, self.dec[0], torch.float16).float(), lat[0], self.comb[0])
        rec = (r.float() - xu).square().mean((1, 2)).sum()
        return ce.mean(-1).sum() + TW * teach + RECON * rec, ce.detach()

    def step(self):
        at = self.counter.view(1)
        m = self.mixs.index_select(0, at)
        # inverse-CDF sampling: uniform mixed with last-loss weights
        c = (self.scores / self.scores.sum(-1, keepdim=True) * m + (1 - m) / self.n).cumsum(-1)
        il = torch.searchsorted(c, torch.rand(M, BATCH, device=self.device) * c[:, -1:]).clamp_max(self.n - 1)
        iu = self.iu.index_select(0, at).squeeze(0)
        self.lr.copy_(self.lrs.index_select(0, at).squeeze(0))
        loss, ce = self.lossfn(self.x[il], self.y[il], self.pool[iu], self.ptar[iu])
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.scores.view(-1).index_copy_(0, (il + self.offs).view(-1), ce.view(-1))
            self.counter.add_(1)

    @torch.no_grad()
    def reset(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(SEED)
        for w in self.enc + self.dec:
            w.copy_(torch.randn(w.shape, generator=g, device=self.device) * w.shape[1] ** -0.5)
        rows = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15]  # weights; biases start at 0
        for p in self.comb:
            p.zero_()
            p[rows] = torch.randn(len(rows), *p.shape[1:], generator=g, device=self.device) * CSTD
        for p in self.beta:
            p.zero_()
        self.gamma.fill_(1.0)
        for p in self.params:
            st = self.opt.state.get(p)
            if st:
                st["exp_avg"].zero_()
                st["exp_avg_sq"].zero_()
                st["step"].zero_()
        self.counter.zero_()
        self.scores.fill_(1.0)
        npool = self.n + self.nq
        per = npool // BATCH
        epochs = math.ceil(STEPS / per)
        u = torch.rand(M, epochs, npool, generator=g, device=self.device).argsort(-1)[:, :, :per * BATCH]
        self.iu.copy_(u.reshape(M, -1, BATCH)[:, :STEPS].transpose(0, 1))
        torch.cuda.manual_seed(SEED + 1)

    def act(self, i, t, b, gm):
        return (t + b) * gm if i == len(self.enc) - 1 else F.relu(t + b)

    @torch.no_grad()
    def predict(self, q):
        """BN statistics averaged over minibatches of training rows, then the clean encoder on q: (M, nq, 10)."""
        g = torch.Generator(device=self.device)
        g.manual_seed(1)
        nb = self.n // BATCH
        h = self.x[torch.randperm(self.n, generator=g, device=self.device)[:nb * BATCH]]
        h = h.reshape(1, nb, BATCH, -1).expand(M, -1, -1, -1)
        q = q.expand(M, -1, -1)
        for i, w in enumerate(self.enc):
            raw = mm(h, w)
            var, mean = torch.var_mean(raw, dim=2, unbiased=False, keepdim=True)
            h = self.act(i, (raw - mean) * torch.rsqrt(var + EPS), self.beta[i][:, None], self.gamma[:, None])
            mu, vv = mean.mean(1), var.mean(1) * BATCH / (BATCH - 1)
            q = self.act(i, (mm(q, w) - mu) * torch.rsqrt(vv + EPS), self.beta[i], self.gamma)
        return q


def ladder_kernel(train_x, train_y, test_x):
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor-cache")
    torch.backends.cuda.matmul.allow_tf32 = False  # teacher in fp32; Ladder may use TF32
    s, transform = teacher(train_x, train_y, test_x)
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        x, q = rownorm(train_x.float()) @ transform, rownorm(test_x.float()) @ transform
        c = (train_x.float() * SCALE).std() / x.std()
        x, q = x * c, q * c
    key = ("ladder", x.shape[0], q.shape[0], str(x.device))
    if key not in CACHE:
        CACHE[key] = Ladder(x.shape[0], q.shape[0], x.device)
    net = CACHE[key]
    with torch.no_grad():
        # pool = train + test; targets: labels, then softened teacher scores
        net.x.copy_(x)
        net.y.copy_(F.one_hot(train_y.long(), 10).float())
        net.pool.copy_(torch.cat((x, q)))
        net.ptar.copy_(torch.cat((net.y, F.softmax(s / TEACH_TAU, 1))))
    net.reset()
    for _ in range(STEPS):
        net.graph.replay()
    with torch.no_grad():
        p = F.softmax(net.predict(q), -1).mean(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    return p.argmax(1)


if __name__ == "__main__":
    import mnist
    print(mnist.score(ladder_kernel, difficulty=5), "ms per call")
