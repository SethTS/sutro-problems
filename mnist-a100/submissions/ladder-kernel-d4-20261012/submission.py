"""MNIST, difficulty 4: a Ladder network on kernel-metric inputs, taught by a kernel model.

Credits: the Ladder of Rasmus et al. 2015 (AMLP combinator of Pezeshki et al. 2016), with the recipe of @SethTS's
Ladder entries (batch 1,000, bottom denoising cost, loss-proportional labelled sampling), in our own plain-PyTorch
version; the kernel-RFM teacher builds on @islamborghini's kernel-pcg entry and Recursive Feature Machines.
Each call trains from scratch:
1. Metric: rows partly normalized in length; twice (on 3,000 rows, then 10,000) fit RBF kernel ridge, take the
   AGOP (mean outer product of its input gradients) and warp the inputs by AGOP^0.75. The data is whitened and
   rotated, so only the labels can say which directions matter.
2. Teacher: kernel ridge by Nystrom-preconditioned CG, stopped early (a regularizer). Test rows with a top-two
   margin >= 0.15 join with their predicted labels; the refit's scores on test rows are soft labels.
3. Student inputs: the rows mapped by AGOP^0.5, a flatter metric than the kernel's.
4. Ladder 60-1024-512-256-10, 560 Adam steps: noisy BN encoder; the decoder rebuilds the clean input from the
   noisy lateral activations. Loss: cross-entropy on labelled train rows, cross-entropy to the targets of an
   unlabelled train + test pool (true labels, or the teacher's for test rows), denoising cost on that pool.
5. Predict with the clean encoder, BN statistics averaged over training minibatches.
test_x is used transductively, without labels. Training is nondeterministic on the GPU, so reruns on one draw
give slightly different predictions.
"""
import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# teacher: RBF exp(-GAMMA |a-b|^2), pseudo-label margin, kernel panel rows, exp shift
GAMMA, RIDGE, RANK, SAMPLES, BLEND, NPOW = 0.02, 0.1, 256, 512, 0.02, 0.6
AGOP_N, M_STEPS, B_STEPS, R_STEPS = (3000, 10000), (16, 16), 14, 16
PSEUDO_CONF, P, NS_ITERS, SHIFT = 0.15, 2000, 15, 3.0
# Ladder: MIX*: loss-sampling schedule; TW, TEACH_TAU: unlabelled-loss weight, teacher temperature
DIMS = (60, 1024, 512, 256, 10)
STEPS, BATCH, LR, BETA1, DECAY_START = 560, 1000, 0.011, 0.8, 2 / 3
NOISE, SCALE, RECON, EPS, CSTD, SEED = 0.25, 0.6, 2000.0, 1e-10, 0.025, 11
MIX, MIX_A, MIX_B, MIX_C = 0.5, 0.1, 0.6, 0.9
TW, TEACH_TAU = 0.5, 0.1
COMPILE = True
F32, H = torch.float32, torch.bfloat16  # activation storage: bf16 (fp16 range fails in the decoder backward)
STATE = {}


# ------------------------------------------------------------------ kernel teacher

def chol(a):
    return torch.linalg.cholesky_ex(a)[0]


def rownorm(x):
    return x * (x.shape[1] ** 0.5 / x.norm(dim=1, keepdim=True).clamp_min(1e-6)) ** NPOW


def aug(a, b, shift=0.0):
    # A @ B = shift - GAMMA |a_i - b_j|^2
    sa = a.square().sum(1, keepdim=True)
    sb = b.square().sum(1, keepdim=True)
    A = torch.cat([a, sa, torch.ones_like(sa)], 1)
    B = torch.cat([(2 * GAMMA) * b, torch.full_like(sb, -GAMMA), shift - GAMMA * sb], 1).T.contiguous()
    return A, B


def kern(a, b):
    A, B = aug(a, b)
    return (A @ B).exp_()


def h3_ops(z):
    # fp16 hi/lo split: one fp16 matmul of Ah Bh + Ah Bl + Al Bh is near-fp32 accurate
    A, B = aug(z, z, SHIFT)
    Ah, Bh = A.half(), B.half()
    Al, Bl = (A - Ah.float()).half(), (B - Bh.float()).half()
    return torch.cat([Ah, Ah, Al], 1), torch.cat([Bh, Bl, Bh], 0)


def panels(A3, B3, pp, rows, diag):
    # upper row panels: rows i:i+pp, columns i:; diagonal = kernel + ridge
    out = []
    for i in rows:
        r = min(pp, A3.shape[0] - i)
        pi = torch.mm(A3[i:i + r], B3[:, i:])
        pi.exp_()
        pi[:, :r].diagonal().fill_(diag)
        out.append(pi)
    return out


class Sym:
    """K[:n, :n] from upper row panels of pp rows"""

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


def upd_eager(ap, d, w, r, nrm, v, vt, mask):
    # preconditioner I - v v^T = lam (u u^T + lam I)^-1 (Woodbury); masked rows stay 0
    if mask is not None:
        ap = ap * mask[:, None]
    a = nrm / (d * ap).sum(0)
    w = w + d * a
    r = r - ap * a
    p = r - v @ (vt @ r)
    nxt = (r * p).sum(0)
    d = p + d * (nxt / nrm)
    return w, r, d, nxt, d.half()


@torch.compile(dynamic=False)
def upd_comp(ap, d, w, r, nrm, v, vt, mask):
    return upd_eager(ap, d, w, r, nrm, v, vt, mask)


def lfac(kmm, jit):
    # Nystrom features: K_nm @ lfac = u with u u^T ~ K
    lm = kmm.clone()
    lm.diagonal().add_(jit)
    eye = torch.eye(len(kmm), device=kmm.device)
    return torch.linalg.solve_triangular(chol(lm), eye, upper=False).T


def pcg(op, t, lam, steps, u, mask=None):
    if mask is not None:
        u = u * mask[:, None]
        t = t * mask[:, None]
    small = u.T @ u
    small.diagonal().add_(lam)
    eye = torch.eye(len(small), device=u.device)
    v = u @ torch.linalg.solve_triangular(chol(small), eye, upper=False).T
    vt = v.T.contiguous()
    w = torch.zeros_like(t)
    r = t.clone()
    d = torch.addmm(r, v, vt @ r, alpha=-1)
    nrm = torch.linalg.vecdot(r, d, dim=0)
    dh = d.half()
    upd = upd_comp if COMPILE else upd_eager
    for _ in range(steps):
        w, r, d, nrm, dh = upd(op.mv(dh), d, w, r, nrm, v, vt, mask)
    return w


def sqrtm(a):
    # Newton-Schulz on [[0, a], [I, 0]]: matmuls only
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
    # score gradients at the first s rows (up to 2 GAMMA), in original coordinates
    s = fac.shape[0]
    wp = torch.cat([w, (w[:, :, None] * z[:, None, :]).reshape(len(z), -1)], 1)
    r = fac.float() @ wp
    c = w.shape[1]
    g = r[:, c:].reshape(s, c, -1) - r[:, :c, None] * z[:s, None, :]
    dim = transform.shape[0]
    g = (g @ transform.T).reshape(-1, dim)
    sq = sqrtm(g.T @ g)

    def fin(m):
        m = (m + m.T) / 2
        m = m * (dim / m.trace().clamp_min(1e-12))
        return chol((1 - BLEND) * m + BLEND * torch.eye(dim, device=z.device))
    return fin(sq @ sqrtm(sq)), fin(sq)  # kernel metric factor (AGOP^0.75), student feature factor (AGOP^0.5)


def kernel_scores(train_x, train_y, test_x):
    c = math.exp(SHIFT)  # stored kernels are c*K
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
            transform, stud = agop(transform, z, w, fac)
        # side stream: test-row panels and Nystrom features, during the train-only fit
        za = torch.cat([x, q]) @ transform
        N = len(za)
        A3, B3 = h3_ops(za)
        pan = panels(A3, B3, P, range(0, n, P), diag)
        km = kern(za, za[:RANK]) * c
        lf = lfac(km[:RANK], jit)
        main = torch.cuda.current_stream()
        ss = torch.cuda.Stream()
        ss.wait_stream(main)
        with torch.cuda.stream(ss):
            pan_t = panels(A3, B3, P, range(n, N, P), diag)
            ua = km @ lf
        opx = Sym(pan, P, n)
        w = pcg(opx, y, lam, B_STEPS, km[:n] @ lf)
        s = opx.cross(w, n, N)
        top = s.topk(2, 1).values
        keep = (top[:, 0] - top[:, 1]) >= PSEUDO_CONF
        mask = torch.cat([torch.ones(n, device=x.device), keep.float()])
        t = F.one_hot(s.argmax(1), 10).float()
        main.wait_stream(ss)
        opn = Sym(pan + pan_t, P, N)
        a = pcg(opn, torch.cat([y, t]), lam, R_STEPS, ua, mask)
        sc = a.abs().amax(0, keepdim=True).clamp_min(1e-30)
        ah = (a / sc).half()
        out = opn.mv(ah)[n:]  # (cK + lam I) a on the test rows -> refit scores c K[n:, :] a
        return (out - lam * ah[n:].float()) * sc, stud


def teacher(train_x, train_y, test_x):
    # first call: two eager runs (compile, warm-up), then one CUDA graph; later calls replay it
    key = ("teacher", tuple(train_x.shape), tuple(test_x.shape), str(train_x.device))
    if key not in STATE:
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                kernel_scores(*static)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = kernel_scores(*static)
        STATE[key] = (g, static, out)
    g, static, out = STATE[key]
    for buf, v in zip(static, (train_x, train_y, test_x)):
        buf.copy_(v)
    g.replay()
    return out[0].clone(), out[1].clone()


# ------------------------------------------------------------------ ladder network

def bn(u):
    uf = u.float()
    var, mean = torch.var_mean(uf, dim=-2, unbiased=False, keepdim=True)
    return ((uf - mean) * torch.rsqrt(var + EPS)).to(u.dtype)


def lrelu(t):
    return F.leaky_relu(t, 0.1)


def dlrelu(t):
    return torch.where(t > 0, 1.0, 0.1)


def combine0(u, lat, p):
    # AMLP per unit; p: W1 0-5, b1 6-7, W2 8-11, b2 12-13, W3 14-15, b3 16
    v = bn(u)
    x2 = v * lat
    a0 = lrelu(v * p[0] + lat * p[2] + x2 * p[4] + p[6])
    a1 = lrelu(v * p[1] + lat * p[3] + x2 * p[5] + p[7])
    c0 = lrelu(a0 * p[8] + a1 * p[10] + p[12])
    c1 = lrelu(a0 * p[9] + a1 * p[11] + p[13])
    return c0 * p[14] + c1 * p[15] + p[16]


class Comb(torch.autograd.Function):
    """per-unit MLP combinator; hand-derived backward recomputes the forward (fp32 statistics and sums)"""

    @staticmethod
    def forward(ctx, u, lat, p32):
        ctx.save_for_backward(u, lat, p32)
        return combine0(u, lat, p32.to(u.dtype))

    @staticmethod
    def backward(ctx, go):
        u, lat, p32 = ctx.saved_tensors
        dt = u.dtype
        p, go, uf = p32.to(dt), go.to(dt), u.float()
        var, mean = torch.var_mean(uf, dim=-2, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + EPS)
        v = ((uf - mean) * rstd).to(dt)
        x2 = v * lat
        a0p = v * p[0] + lat * p[2] + x2 * p[4] + p[6]
        a1p = v * p[1] + lat * p[3] + x2 * p[5] + p[7]
        a0, a1 = lrelu(a0p), lrelu(a1p)
        c0p = a0 * p[8] + a1 * p[10] + p[12]
        c1p = a0 * p[9] + a1 * p[11] + p[13]
        c0, c1 = lrelu(c0p), lrelu(c1p)
        gc0 = go * p[14] * dlrelu(c0p)
        gc1 = go * p[15] * dlrelu(c1p)
        ga0 = (gc0 * p[8] + gc1 * p[9]) * dlrelu(a0p)
        ga1 = (gc0 * p[10] + gc1 * p[11]) * dlrelu(a1p)
        gv = (ga0 * (p[0] + p[4] * lat) + ga1 * (p[1] + p[5] * lat)).float()
        glat = ga0 * (p[2] + p[4] * v) + ga1 * (p[3] + p[5] * v)
        vf = v.float()
        gu = rstd * (gv - gv.mean(-2, keepdim=True) - vf * (gv * vf).mean(-2, keepdim=True))
        r = lambda t: t.float().sum(-2, keepdim=True)
        gp = torch.stack((r(ga0 * v), r(ga1 * v), r(ga0 * lat), r(ga1 * lat), r(ga0 * x2), r(ga1 * x2), r(ga0), r(ga1),
                          r(gc0 * a0), r(gc1 * a0), r(gc0 * a1), r(gc1 * a1), r(gc0), r(gc1), r(go * c0), r(go * c1),
                          r(go)))
        return gu.to(dt), glat.to(lat.dtype), gp


def mm16(a, w):
    s = a.shape
    return torch.bmm(a.reshape(1, -1, s[-1]), w.to(H)).reshape(*s[:-1], w.shape[-1])


class Ladder:
    def __init__(self, n, nq, device):
        self.n, self.nq, self.device = n, nq, device
        zeros = lambda *s: torch.zeros(*s, device=device)
        self.enc = [zeros(1, a, b) for a, b in zip(DIMS[:-1], DIMS[1:])]
        self.dec = [zeros(1, b, a) for a, b in zip(DIMS[:-1], DIMS[1:])]
        self.beta = [zeros(1, 1, b) for b in DIMS[1:]]
        self.gamma = zeros(1, 1, DIMS[-1])
        self.comb = [zeros(17, 1, 1, s) for s in DIMS]
        self.params = self.enc + self.dec + self.beta + [self.gamma] + self.comb
        for p in self.params:
            p.requires_grad_(True)
        self.x = zeros(n, DIMS[0])
        self.y = zeros(n, 10)
        self.pool = zeros(n + nq, DIMS[0])
        self.ptar = zeros(n + nq, 10)
        self.scores = zeros(1, n)
        self.counter = torch.zeros((), dtype=torch.long, device=device)
        self.iu = torch.zeros(STEPS, 1, BATCH, dtype=torch.long, device=device)
        frac = torch.arange(STEPS, device=device, dtype=torch.float32) / STEPS
        self.lrs = LR * ((1 - frac) / (1 - DECAY_START)).clamp(max=1.0)
        ramp = ((MIX_C - frac) / (MIX_C - MIX_B)).clamp(0, 1)
        self.mixs = MIX * torch.where(frac < MIX_A, 0.0, torch.where(frac < MIX_B, 1.0, ramp))
        self.lr = torch.tensor(LR, device=device)
        self.opt = torch.optim.Adam(self.params, lr=self.lr, betas=(BETA1, 0.999), fused=True, capturable=True)
        self.lossfn = torch.compile(self.loss, dynamic=False) if COMPILE else self.loss
        self.reset()
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
        # labelled and unlabelled rows side by side, separate BN statistics
        h = torch.stack((xl, xu), 1)
        h = (h + NOISE * torch.randn_like(h)).to(H)
        lat = [h[:, 1]]
        top = len(self.enc) - 1
        for i, w in enumerate(self.enc):
            z = bn(mm16(h, w))
            z = z + NOISE * torch.randn_like(z)
            lat.append(z[:, 1])
            b = self.beta[i][:, None].to(H)
            h = (z + b) * self.gamma[:, None].to(H) if i == top else F.relu(z + b)
        hf = h.float()
        ce = -(yl * F.log_softmax(hf[:, 0], -1)).sum(-1)
        teach = -(tu * F.log_softmax(hf[:, 1], -1)).sum(-1).mean(-1).sum()
        # decoder on the unlabelled rows; denoising cost at the input only
        r = Comb.apply(F.softmax(hf[:, 1], -1).to(H), lat[-1], self.comb[-1])
        for i in range(top, -1, -1):
            r = Comb.apply(mm16(r, self.dec[i]), lat[i], self.comb[i])
        rec = (r.float() - xu).square().mean((1, 2)).sum()
        return ce.mean(-1).sum() + TW * teach + RECON * rec, ce.detach()

    def step(self):
        # labelled rows: inverse-CDF draw from a uniform / last-loss mix
        at = self.counter.view(1)
        m = self.mixs.index_select(0, at)
        c = (self.scores / self.scores.sum(-1, keepdim=True) * m + (1 - m) / self.n).cumsum(-1)
        il = torch.searchsorted(c, torch.rand(1, BATCH, device=self.device) * c[:, -1:]).clamp_max(self.n - 1)
        iu = self.iu.index_select(0, at).squeeze(0)
        self.lr.copy_(self.lrs.index_select(0, at).squeeze(0))
        loss, ce = self.lossfn(self.x[il], self.y[il], self.pool[iu], self.ptar[iu])
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.scores.view(-1).index_copy_(0, il.view(-1), ce.view(-1))
            self.counter.add_(1)

    @torch.no_grad()
    def reset(self):
        g = torch.Generator(device=self.device)
        g.manual_seed(SEED)
        for w in self.enc + self.dec:
            w.copy_(torch.randn(w.shape, generator=g, device=self.device) * w.shape[1] ** -0.5)
        rows = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15]
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
        u = torch.rand(1, epochs, npool, generator=g, device=self.device).argsort(-1)[:, :, :per * BATCH]
        self.iu.copy_(u.reshape(1, -1, BATCH)[:, :STEPS].transpose(0, 1))
        torch.cuda.manual_seed(SEED + 1)

    def run(self):
        for _ in range(STEPS):
            self.graph.replay()

    @torch.no_grad()
    def predict(self, q):
        """BN statistics averaged over minibatches of training rows, then the clean encoder on q"""
        k, enc, beta, gamma = len(self.enc), self.enc, self.beta, self.gamma
        g = torch.Generator(device=self.device)
        g.manual_seed(1)
        nb = self.n // BATCH
        h = self.x[torch.randperm(self.n, generator=g, device=self.device)[:nb * BATCH]].reshape(1, nb, BATCH, -1)
        q = q[None]
        for i, w in enumerate(enc):
            raw = torch.matmul(h, w[:, None])
            var, mean = torch.var_mean(raw, dim=2, unbiased=False, keepdim=True)
            mu, vv = mean.mean(1), var.mean(1) * BATCH / (BATCH - 1)
            act = lambda t, b: (t + b) * gamma if i == k - 1 else F.relu(t + b)
            h = act((raw - mean) * torch.rsqrt(var + EPS), beta[i][:, None])
            q = act((q @ w - mu) * torch.rsqrt(vv + EPS), beta[i])
        return q[0]


def ladder_kernel(train_x, train_y, test_x):
    first = "ladder" not in STATE  # warm-up call: compile, capture graphs
    torch.backends.cuda.matmul.allow_tf32 = False
    s, sf = teacher(train_x, train_y, test_x)  # fp32 teacher, TF32 student
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.no_grad():
        x, q = rownorm(train_x.float()) @ sf, rownorm(test_x.float()) @ sf
        c = (train_x.float() * SCALE).std() / x.std()
        x, q = x * c, q * c
    if first:
        STATE["ladder"] = Ladder(x.shape[0], q.shape[0], x.device)
    net = STATE["ladder"]
    with torch.no_grad():
        net.x.copy_(x)
        net.y.copy_(F.one_hot(train_y.long(), 10).float())
        net.pool.copy_(torch.cat((x, q)))
        net.ptar.copy_(torch.cat((net.y, F.softmax(s / TEACH_TAU, 1))))
    net.reset()
    net.run()
    with torch.no_grad():
        out = net.predict(q).argmax(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if first:
        import gc
        gc.collect()
        gc.freeze()  # keep the compile machinery's objects out of later collections (wall clock)
    return out


if __name__ == "__main__":
    import mnist
    print(mnist.score(ladder_kernel, difficulty=4), "ms per call")
