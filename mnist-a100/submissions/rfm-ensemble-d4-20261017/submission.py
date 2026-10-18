"""D4 candidate: five independently sampled RFM metrics with three metric updates, transductive kernel ridge,
local affine-hull correction, and averaged class probabilities. No backpropagation.
All fits reset on every call. Only model predictions label query hull vertices.
Credits: islamborghini kernel-PCG; Radhakrishnan et al. RFM; spartanhaden's
Ladder/kernel teacher and RFM/HKNN entry; Vincent & Bengio HKNN.
The derivative-penalty variants are retained separately as research evidence.
"""
"Frozen analytic RFM teacher derived from spartanhaden's Ladder/kernel entry.\nNo neural-network training. Returns scores, metric coordinates, kernel coefficients.\nCredits: islamborghini kernel-PCG; Radhakrishnan et al. RFM; spartanhaden teacher.\n"
import math
import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
(GAMMA, RIDGE, RANK, SAMPLES, BLEND, NPOW) = (0.02, 0.1, 256, 2048, 0.02, 0.6)
(AGOP_N, M_STEPS, B_STEPS, R_STEPS) = ((3000, 6000, 10000), (12, 12, 12), 13, 16)
(PSEUDO_CONF, P, NS_ITERS, SHIFT) = (0.15, 2000, 7, 3.0)
F32 = torch.float32
STATE = {}

def chol(a):
    return torch.linalg.cholesky_ex(a)[0]

def rownorm(x):
    return x * (x.shape[1] ** 0.5 / x.norm(dim=1, keepdim=True).clamp_min(1e-06)) ** NPOW

def aug(a, b, shift=0.0):
    sa = a.square().sum(1, keepdim=True)
    sb = b.square().sum(1, keepdim=True)
    A = torch.cat([a, sa, torch.ones_like(sa)], 1)
    B = torch.cat([2 * GAMMA * b, torch.full_like(sb, -GAMMA), shift - GAMMA * sb], 1).T.contiguous()
    return (A, B)

def kern(a, b):
    (A, B) = aug(a, b)
    return (A @ B).exp_()

def h3_ops(z):
    (A, B) = aug(z, z, SHIFT)
    (Ah, Bh) = (A.half(), B.half())
    (Al, Bl) = ((A - Ah.float()).half(), (B - Bh.float()).half())
    return (torch.cat([Ah, Ah, Al], 1), torch.cat([Bh, Bl, Bh], 0))

def panels(A3, B3, pp, rows, diag):
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
        (self.pans, self.pp, self.n) = (pans, pp, n)

    def mv(self, dh):
        (pp, n) = (self.pp, self.n)
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
        for (j, pj) in enumerate(self.pans):
            a = j * self.pp
            if a >= lo:
                break
            if j == 0:
                torch.mm(pj[:, lo - a:hi - a].T, wh[a:a + pj.shape[0]], out_dtype=F32, out=y)
            else:
                torch.addmm(y, pj[:, lo - a:hi - a].T, wh[a:a + pj.shape[0]], out_dtype=F32, out=y)
        return y * s

def upd_eager(ap, d, w, r, nrm, v, vt, mask):
    if mask is not None:
        ap = ap * mask[:, None]
    a = nrm / (d * ap).sum(0)
    w = w + d * a
    r = r - ap * a
    p = r - v @ (vt @ r)
    nxt = (r * p).sum(0)
    d = p + d * (nxt / nrm)
    return (w, r, d, nxt, d.half())

def lfac(kmm, jit):
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
    upd = upd_eager
    for _ in range(steps):
        (w, r, d, nrm, dh) = upd(op.mv(dh), d, w, r, nrm, v, vt, mask)
    return w

def sqrtm(a):
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
    return (fin(sq @ sqrtm(sq)), fin(sq))

def kernel_scores(train_x, train_y, test_x):
    c = math.exp(SHIFT)
    (lam, jit, diag) = (RIDGE * c, 0.0001 * c, c * (1.0 + RIDGE))
    with torch.no_grad():
        (x, q) = (rownorm(train_x.float()), rownorm(test_x.float()))
        n = len(x)
        y = F.one_hot(train_y.long(), 10).float()
        transform = torch.eye(x.shape[1], device=x.device)
        for (m, st) in zip(AGOP_N, M_STEPS):
            z = x[:m] @ transform
            km = kern(z, z[:RANK]) * c
            pm = P if m % P == 0 else m
            (A3, B3) = h3_ops(z)
            pan = panels(A3, B3, pm, range(0, m, pm), diag)
            w = pcg(Sym(pan, pm, m), y[:m], lam, st, km @ lfac(km[:RANK], jit))
            fac = pan[0][:SAMPLES].clone()
            fac.diagonal().fill_(c)
            del pan
            (transform, stud) = agop(transform, z, w, fac)
        za = torch.cat([x, q]) @ transform
        N = len(za)
        (A3, B3) = h3_ops(za)
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
        keep = top[:, 0] - top[:, 1] >= PSEUDO_CONF
        mask = torch.cat([torch.ones(n, device=x.device), keep.float()])
        t = F.one_hot(s.argmax(1), 10).float()
        main.wait_stream(ss)
        opn = Sym(pan + pan_t, P, N)
        a = pcg(opn, torch.cat([y, t]), lam, R_STEPS, ua, mask)
        sc = a.abs().amax(0, keepdim=True).clamp_min(1e-30)
        ah = (a / sc).half()
        out = opn.mv(ah)[n:]
        return ((out - lam * ah[n:].float()) * sc, za)
(HK_K, HK_LAM, HK_T, HK_W, HK_C, HK_M) = (70, 8.0, 2.0, 0.15, 3, 2048)

def hknn(zq, zv, yv, valid, cls):
    """HKNN (Vincent & Bengio 2002), transductive: for each query and each of its classes cls (nq, C), the
    squared distance to the ridge-regularised affine hull of the HK_K nearest valid vertices of that class,
    searched among the query's HK_M nearest vertices (fp16 distances). Static shapes (graph-safe): a class
    with fewer members there repeats its nearest one; a class with none gets distance 1e30. Primal closed
    form: residual = lam (V^T V + lam I)^-1 r, one batched Cholesky."""
    (nq, C) = cls.shape
    dim = zq.shape[1]
    D = (zq.square().sum(1, keepdim=True) + zv.square().sum(1)[None] - 2 * zq @ zv.T).clamp_min_(0).half()
    (cd, ci) = D.topk(HK_M, 1, largest=False)
    ok = valid[ci][:, None, :] & (yv[ci][:, None, :] == cls[:, :, None])
    (sv, si) = cd[:, None, :].masked_fill(~ok, float('inf')).topk(HK_K, 2, largest=False)
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
    return out.masked_fill(bad[:, :, 0], 1e+30)

def hk_probs(d, t):
    """class probabilities from hull distances; the temperature scales with the mean nearest distance"""
    dn = d - d.min(1, keepdim=True).values
    return torch.softmax(-dn / (t * d.min(1).values.mean().clamp_min(1e-12)), 1)

def correct(scores, z, train_y, weight=3.012830538):
    n = len(train_y)
    top = scores.topk(2, 1).values
    margin = top[:, 0] - top[:, 1]
    idx = margin.topk(512, largest=False).indices
    yall = torch.cat([train_y.long(), scores.argmax(1)])
    valid = torch.cat([torch.ones(n, dtype=torch.bool, device=z.device), margin >= 0.2])
    su = scores[idx]
    cls = su.topk(HK_C, 1).indices
    pr = hk_probs(hknn(z[n:][idx], z, yall, valid, cls), HK_T)
    out = scores.clone()
    out[idx] = su + weight * torch.zeros_like(su).scatter_(1, cls, pr) * (margin[idx] < 0.2)[:, None]
    return out

def run(train_x, train_y, test_x):
    global GAMMA, NPOW
    probs = []
    settings = ((0.02, 0.6),) * 5 + ((0.015, 0.6), (0.025, 0.6), (0.02, 0.4), (0.02, 0.8))
    with torch.no_grad():
        for member in range(5):
            (GAMMA, NPOW) = settings[member]
            offset = member % 5 * (len(train_x) // 5)
            (rx, ry) = (train_x.roll(offset, 0), train_y.roll(offset, 0))
            (scores, z) = kernel_scores(rx, ry, test_x)
            hs = correct(scores, z, ry)
            mixed = scores + 1.0 * (hs - scores)
            probs.append(torch.softmax(mixed / 0.3, 1))
        return torch.stack(probs).mean(0).argmax(1)

def kernel_ensemble(train_x, train_y, test_x):
    key = (tuple(train_x.shape), tuple(test_x.shape), str(train_x.device))
    if key not in STATE:
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
        STATE[key] = (graph, static, out)
    (graph, static, out) = STATE[key]
    for (buf, v) in zip(static, (train_x, train_y, test_x)):
        buf.copy_(v)
    graph.replay()
    return out.clone()
