"""MNIST difficulty 3, fast: RBF kernel ridge in a learned metric, then HKNN on the low-margin rows. No backprop.

Builds on @islamborghini's kernel-pcg (RBF kernel ridge, an AGOP metric, Nystrom-preconditioned CG) and HKNN
local affine hulls (Vincent & Bengio 2002).
No gradient descent: closed-form linear algebra, a few CG steps and analytic gradients.

Steps, per call:
 1. Partial row normalisation pulls every row's norm part of the way toward sqrt(60).
 2. Metric learning, twice. Fit RBF kernel ridge to the one-hot labels (first on 3000 training rows in the
    plain metric, then on 6000 in the updated one; 12 CG steps each). Take the AGOP of the fitted scores (the
    average outer product of their input gradients at 512 training rows) as M. Use ~M^0.75, trace-normalised
    with a 2% identity blend, as the next Mahalanobis metric. The 0.75 power comes from two 7-step
    Newton-Schulz square roots; at 7 steps the weakest directions get roughly M^1 instead (a soft floor).
 3. Base fit: kernel ridge on all 10000 training rows in the final metric (13 CG steps), then scores for
    the test rows.
 4. HKNN on the 512 lowest-margin test rows. For each row's 3 highest-scoring classes, take the distance to
    the ridge-regularised affine hull of that class's 70 nearest vertices (training rows plus test rows whose
    score margin is >= 0.2, carrying their predicted label), searched among the row's 2048 nearest vertices.
    Turn the distances into class probabilities and add 0.15x them to the kernel scores (in kernel units).
    Only rows with margin < 0.2 among the 512 can change label; every other row keeps its base-fit label.
Every fit stops after a fixed, small number of CG steps; the early stop also regularises the fit.
test_x is used without labels: per-row normalisation and model-made labels for the HKNN vertices.

There is no pseudo-label refit of the kernel model (with the scores in kernel units, HKNN carries the
low-margin rows) and no torch.compile (all eager ops), so there is no compile warm-up.

Speed: kernel exponents come from one fp16 GEMM with a hi/lo operand split. Symmetric kernels are stored in
half as fp16 row panels; matvecs alternate the panel order so each step starts on the panels still in L2.
Test-row scores are built in L2-sized chunks and never stored. Each fit's preconditioner (a latency-bound
chain of small Cholesky solves) runs on a side stream while its kernel panels are built. The whole method
runs as one CUDA graph.
Plain PyTorch only; no hand-written kernels.

    python mnist.py submission.py:rfm_d3 --difficulty 3
"""

import gc
import math

import torch
import torch.nn.functional as F

# fp32 GEMMs stay true fp32 (no TF32); fp16 GEMMs accumulate in fp32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# RBF width, ridge, Nystrom rank, identity blend, row-norm power
GAMMA, RIDGE, RANK, BLEND, NPOW = 0.02, 0.1, 256, 0.02, 0.6
FIT_N, FIT_STEPS, SAMPLES = (3000, 6000), (12, 12), 512  # metric fits: training rows, CG steps; AGOP rows
B_STEPS = 13  # CG steps in the base fit
CONF, NU = 0.2, 512  # score margin of a confident row; number of low-margin rows revisited
HK_K, HK_LAM, HK_T, HK_W, HK_C, HK_M = 70, 8.0, 2.0, 0.15, 3, 2048  # HKNN: neighbours, ridge, temperature,
#                                                       weight, classes per row, candidate vertices per row
# Rows per kernel panel, rows per test-score chunk, Newton-Schulz iterations, exponent shift (kernels are
# stored as e^SHIFT * K so that small kernel values stay well inside the fp16 range)
P, CH, NS_ITERS, SHIFT = 2000, 1000, 7, 3.0
F32 = torch.float32
_GRAPHS = {}


def chol(a):
    return torch.linalg.cholesky_ex(a)[0]


def rownorm(x):
    return x * (x.shape[1] ** 0.5 / x.norm(dim=1, keepdim=True).clamp_min(1e-6)) ** NPOW


def aug(a, b, shift=0.0):
    # [a, |a|^2, 1] @ [2g b; -g; shift - g|b|^2] = shift - g |a - b|^2: the RBF exponent as one GEMM
    sa = a.square().sum(1, keepdim=True)
    sb = b.square().sum(1, keepdim=True)
    A = torch.cat([a, sa, torch.ones_like(sa)], 1)
    B = torch.cat([(2 * GAMMA) * b, torch.full_like(sb, -GAMMA), shift - GAMMA * sb], 1).T.contiguous()
    return A, B


def kern(a, b):
    A, B = aug(a, b)
    return (A @ B).exp_()


def h3_ops(z):
    # hi/lo fp16 split: A B ~= Ah Bh + Ah Bl + Al Bh (Al Bl dropped), as one fp16 GEMM [Ah Ah Al] @ [Bh; Bl; Bh].
    # The exponent's terms largely cancel, so fp16 rounding of the operands would dominate; the split removes it.
    A, B = aug(z, z, SHIFT)
    Ah, Bh = A.half(), B.half()
    Al, Bl = (A - Ah.float()).half(), (B - Bh.float()).half()
    return torch.cat([Ah, Ah, Al], 1), torch.cat([Bh, Bl, Bh], 0)


def panels(A3, B3, n, diag):
    # fp16 kernel row panels K[i:i+P, i:n] (upper block triangle); the diagonal holds the ridge term too
    pp = P if n % P == 0 else n
    out = []
    for i in range(0, n, pp):
        pi = torch.mm(A3[i:i + pp], B3[:, i:n])
        pi.exp_()
        pi[:, :pp].diagonal().fill_(diag)
        out.append(pi)
    return out, pp


class Sym:
    """Symmetric K[:n, :n] stored as upper row panels of pp rows. Products read each panel twice (as itself
    and transposed), in fp16 with fp32 output, in forward or reverse panel order."""

    def __init__(self, pans, pp, n):
        self.pans, self.pp, self.n = pans, pp, n

    def mv(self, dh, rev):
        pp, n = self.pp, self.n
        y = torch.zeros(n, dh.shape[1], device=dh.device)
        js = range(len(self.pans))
        for j in (reversed(js) if rev else js):
            a = j * pp
            pj = self.pans[j]
            e = a + pj.shape[0]
            torch.addmm(y[a:e], pj, dh[a:n], out_dtype=F32, out=y[a:e])
            if e < n:
                torch.addmm(y[e:n], pj[:, e - a:].T, dh[a:e], out_dtype=F32, out=y[e:n])
        return y


def cross(A3, B3, n, w):
    """c K[n:, :n] w, built in CH-row chunks (each chunk stays in L2) and never stored; w rescaled into fp16"""
    s = w.abs().amax(0, keepdim=True).clamp_min(1e-30)
    wh = (w / s).half()
    out = torch.empty(len(A3) - n, w.shape[1], device=w.device)
    for i in range(n, len(A3), CH):
        e = torch.mm(A3[i:i + CH], B3[:, :n]).exp_()
        torch.mm(e, wh, out_dtype=F32, out=out[i - n:i - n + CH])
    return out * s


def lfac(kmm, jit):
    """L^-T for the Nystrom landmarks, with L L^T = K_mm + jit I"""
    lm = kmm.clone()
    lm.diagonal().add_(jit)
    eye = torch.eye(len(kmm), device=kmm.device)
    return torch.linalg.solve_triangular(chol(lm), eye, upper=False).T


def precond(z, lam, jit, c):
    """Nystrom preconditioner factor v for the kernel on the rows z (landmarks: the first RANK rows).
    With u = Km L^-T (L L^T = K_mm + jit I), K ~= u u^T; by Woodbury the preconditioner is (I - v v^T) up to
    a scale, with v = u chol(u^T u + lam I)^-T."""
    km = kern(z, z[:RANK]) * c
    u = km @ lfac(km[:RANK], jit)
    small = u.T @ u
    small.diagonal().add_(lam)
    eye = torch.eye(len(small), device=u.device)
    v = u @ torch.linalg.solve_triangular(chol(small), eye, upper=False).T
    return v, v.T.contiguous()


def pcg(op, t, steps, v, vt):
    """Nystrom-PCG for (K + lam I) w = t; op.mv(dh) = (K + lam I) d; preconditioner I - v v^T. A fixed,
    small number of steps (early stopping). The vector update is a handful of fused eager ops."""
    w = torch.zeros_like(t)
    r = t.clone()
    d = torch.addmm(r, v, vt @ r, alpha=-1)
    nrm = torch.linalg.vecdot(r, d, dim=0)
    for k in range(steps):
        ap = op.mv(d.half(), k % 2 == 1)
        a = nrm / torch.linalg.vecdot(d, ap, dim=0)
        w.addcmul_(d, a)
        r.addcmul_(ap, a, value=-1)
        p = torch.addmm(r, v, vt @ r, alpha=-1)
        nxt = torch.linalg.vecdot(r, p, dim=0)
        d = torch.addcmul(p, d, nxt / nrm)
        nrm = nxt
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
    """Factor L (L L^T = blend(M^0.75)) of the AGOP M of f = sum_j w_j k(., z_j) at the first SAMPLES
    rows. fac holds those rows' kernel values (fp16). The RBF gradient at x is proportional to
    sum_j w_j k(x, z_j) (z_j - x); it is mapped back to input coordinates through transform."""
    s = fac.shape[0]
    wp = torch.cat([w, (w[:, :, None] * z[:, None, :]).reshape(len(z), -1)], 1)
    sc = wp.abs().amax(0, keepdim=True).clamp_min(1e-30)  # columns rescaled into fp16 range
    r = torch.mm(fac, (wp / sc).half(), out_dtype=F32) * sc
    c = w.shape[1]
    g = r[:, c:].reshape(s, c, -1) - r[:, :c, None] * z[:s, None, :]
    dim = transform.shape[0]
    g = (g @ transform.T).reshape(-1, dim)
    m = g.T @ g
    sq = sqrtm(m)
    m = sq @ sqrtm(sq)  # M^0.5 M^0.25 = M^0.75 (with the soft floor of 7-step Newton-Schulz)
    m = (m + m.T) / 2
    m = m * (dim / m.trace().clamp_min(1e-12))
    m = (1 - BLEND) * m + BLEND * torch.eye(dim, device=z.device)
    return chol(m)


def hknn(zq, zv, yv, valid, cls):
    """HKNN (Vincent & Bengio 2002), transductive: for each query and each of its classes cls (nq, C), the
    squared distance to the ridge-regularised affine hull of the HK_K nearest valid vertices of that class,
    searched among the query's HK_M nearest vertices (fp16 distances). Static shapes (graph-safe): a class
    with fewer members there repeats its nearest one; a class with none gets distance 1e30. Primal closed
    form: residual = lam (V^T V + lam I)^-1 r, one batched Cholesky."""
    nq, C = cls.shape
    dim = zq.shape[1]
    D = (zq.square().sum(1, keepdim=True) + zv.square().sum(1)[None] - 2 * zq @ zv.T).clamp_min_(0).half()
    cd, ci = D.topk(HK_M, 1, largest=False)
    ok = valid[ci][:, None, :] & (yv[ci][:, None, :] == cls[:, :, None])  # (nq, C, HK_M)
    sv, si = cd[:, None, :].masked_fill(~ok, float("inf")).topk(HK_K, 2, largest=False)
    nb = ci.gather(1, si.reshape(nq, -1)).reshape(nq, C, HK_K)
    bad = torch.isinf(sv)
    nb = torch.where(bad, nb[:, :, :1], nb)
    nbz = zv[nb].reshape(nq * C, HK_K, dim)
    m = nbz.mean(1, keepdim=True)
    v = nbz - m
    r = (zq[:, None, :].expand(nq, C, dim).reshape(nq * C, 1, dim) - m).transpose(1, 2)
    eye = torch.eye(dim, device=zq.device) * HK_LAM
    lc = torch.linalg.cholesky_ex(torch.baddbmm(eye[None], v.transpose(1, 2), v))[0]
    out = (torch.cholesky_solve(r, lc).square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)
    return out.masked_fill(bad[:, :, 0], 1e30)


def hk_probs(d, t):
    """class probabilities from hull distances; the temperature scales with the mean nearest distance"""
    dn = d - d.min(1, keepdim=True).values
    return torch.softmax(-dn / (t * d.min(1).values.mean().clamp_min(1e-12)), 1)


def run(train_x, train_y, test_x):
    c = math.exp(SHIFT)  # stored kernels are c*K, so the ridge and the Nystrom jitter scale by c too
    lam, jit, diag = RIDGE * c, 1e-4 * c, c * (1.0 + RIDGE)
    with torch.no_grad():
        x, q = rownorm(train_x.float()), rownorm(test_x.float())
        n = len(x)
        y = F.one_hot(train_y.long(), 10).float()
        transform = torch.eye(x.shape[1], device=x.device)
        # two metric updates: fit on the first m training rows in the current metric, AGOP of that fit
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        for m, st in zip(FIT_N, FIT_STEPS):
            z = x[:m] @ transform
            side.wait_stream(main)
            with torch.cuda.stream(side):  # preconditioner (latency-bound) concurrently with the panels
                v, vt = precond(z, lam, jit, c)
            A3, B3 = h3_ops(z)
            pan, pp = panels(A3, B3, m, diag)
            main.wait_stream(side)
            w = pcg(Sym(pan, pp, m), y[:m], st, v, vt)
            fac = pan[0][:SAMPLES].clone()
            fac.diagonal().fill_(c)  # kernel values without the ridge term
            del pan
            transform = agop(transform, z, w, fac)
        # base fit on all training rows in the final metric, then scores for the test rows
        za = torch.cat([x, q]) @ transform
        side.wait_stream(main)
        with torch.cuda.stream(side):
            v, vt = precond(za[:n], lam, jit, c)
        A3, B3 = h3_ops(za)
        pan, pp = panels(A3, B3, n, diag)
        main.wait_stream(side)
        w = pcg(Sym(pan, pp, n), y, B_STEPS, v, vt)
        del pan
        s = cross(A3, B3, n, w)
        top = s.topk(2, 1).values
        marg = top[:, 0] - top[:, 1]
        keep = marg >= CONF
        lab = s.argmax(1)
        # the NU lowest-margin test rows: kernel scores (in K units) + weighted HKNN probabilities
        uidx = marg.topk(NU, largest=False).indices
        su = s[uidx] / c
        cls = su.topk(HK_C, 1).indices
        yall = torch.cat([train_y.long(), lab])
        valid = torch.cat([torch.ones(n, dtype=torch.bool, device=x.device), keep])
        pr = hk_probs(hknn(za[n:][uidx], za, yall, valid, cls), HK_T)
        su = su + HK_W * torch.zeros_like(su).scatter_(1, cls, pr)
        res = lab.clone()
        res[uidx] = torch.where(keep[uidx], lab[uidx], su.argmax(1))
        return res


def rfm_d3(train_x, train_y, test_x):
    """Entry point. The first call for an input shape runs run() eagerly twice and captures it as one CUDA
    graph; later calls copy their inputs into the graph's buffers and replay it (the whole fit reruns)."""
    key = (tuple(train_x.shape), tuple(test_x.shape), train_x.dtype, train_y.dtype, str(train_x.device))
    if key not in _GRAPHS:
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run(*static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(*static)
        _GRAPHS[key] = (graph, static, out)
        gc.collect()
        gc.freeze()  # keep warm-up objects out of every later collection (wall clock)
    graph, static, out = _GRAPHS[key]
    static[0].copy_(train_x)
    static[1].copy_(train_y)
    static[2].copy_(test_x)
    graph.replay()
    return out.clone()


if __name__ == "__main__":
    import mnist
    print(mnist.score(rfm_d3, difficulty=3), "ms per call")
