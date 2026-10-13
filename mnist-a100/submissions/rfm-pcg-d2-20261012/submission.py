"""MNIST, difficulty 2: RBF kernel ridge regression with one learned metric, solved by preconditioned CG.
No neural network and no backprop.

This builds on @islamborghini's kernel-pcg entry (sutro-problems mnist-a100, 2026-09-29): RBF kernel ridge with
one AGOP / Recursive-Feature-Machine metric update (Radhakrishnan et al., arXiv 2212.13881), solved by
Nystrom-preconditioned conjugate gradients. This file re-tunes that method and rebuilds it in plain PyTorch.

Every call fits from scratch on the labels it is given:
1. Scale every train and test row to norm sqrt(60), so that one kernel width suits all rows.
2. Learn a metric (the RFM step). The inputs are whitened and randomly rotated, so only the labels say which
   directions matter. Fit a small kernel ridge model on 2,000 train rows (16 plain CG steps), take the gradients
   of its 10 class scores at 768 of those rows, and sum their outer products into a 60 x 60 matrix M (the AGOP).
   Scale M to trace 0.98 * 60, add 0.02 * I, and map every row to z = x L, where M = L L'. Plain distances
   between z rows are then M-distances between x rows.
3. Fit kernel ridge (K + 0.1 I) w = Y on all 10,000 rows in z space, as one CG solve per class. A rank-512
   Nystrom approximation of K (512 landmark rows, inverted by the Woodbury identity) preconditions it. The
   solve takes only 5 steps, and the last step is damped (BETA): stopping early acts as a regularizer.
4. Predict the argmax of K(test, train) w.

Speed. The 10,000 x 10,000 kernel is never formed. Each product K v runs through PyTorch's fp16 flash-attention
kernel: with q = [sqrt(2g) z, 1] and k = [sqrt(2g) z', -g|z'|^2], softmax(q k') v times exp(logsumexp - g|z|^2)
equals K v. The preconditioner is built on a side stream while the main stream prepares the flash operands, and
its Cholesky factor is inverted explicitly so that applying it costs only GEMMs. The first call for an input
shape runs the fit eagerly, then records the whole fit as one CUDA graph; later calls copy their inputs into the
recorded buffers and replay it, so every call still refits from scratch. There is no torch.compile, so that
warm-up stays under about a second: the scorer stages inputs through a shared mmap of a /tmp file, and after a
pause of seconds its first writes page-fault, which the timed call would pay for.

test_x is used only for per-row scaling and prediction: not transductively, and no test labels. Predictions
are deterministic for given inputs.

    python mnist.py mnist_d2_rfm_pcg.py:classify --difficulty 2
"""
import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False

GAMMA, RIDGE, STEPS, RANK, JITTER = 0.0175, 0.1, 5, 512, 1e-4
PRE_RIDGE = 0.17   # the preconditioner uses a larger ridge than the system itself
BETA = -0.3        # final w = w5 + BETA (w5 - w4): the last CG step is shortened to 0.7 of its length
# Metric step: subset size, plain-CG steps and ridge for its sub-fit, gradient sample count, identity blend.
SUB, SUB_STEPS, SUB_RIDGE, SAMPLES, BLEND = 2000, 16, 0.1, 768, 0.02
TPAR = 4           # streams used to invert the preconditioner's Cholesky factor
STATE = {}         # CUDA streams and the captured graph, per input shape (no learned state)


def aug(a, g, left):
    """Augmented rows whose inner products give -g |a - b|^2: [sqrt(2g) a, -g|a|^2, 1] . [sqrt(2g) b, 1, -g|b|^2]."""
    n2 = a.square().sum(1, keepdim=True) * (-g)
    one = torch.ones_like(n2)
    s = a * math.sqrt(2 * g)
    return torch.cat([s, n2, one], 1) if left else torch.cat([s, one, n2], 1)


def aug16(a, g, left):
    """aug() in fp16, zero-padded to a multiple of 8 columns for the tensor cores."""
    h = aug(a, g, left).half()
    return F.pad(h, (0, (-h.shape[1]) % 8))


def kernel32(a, b, g):
    """RBF kernel exp(-g |a - b|^2) in fp32 (one GEMM, then an in-place exp)."""
    return (aug(a, g, True) @ aug(b, g, False).T).exp_()


def fexp(a, b):
    """exp(a @ b.T) in fp16 (cuBLAS GEMM, then an in-place exp)."""
    return (a @ b.T).exp_()


def stream(name, prio=0):
    """A named CUDA stream, created once and reused across calls."""
    if name not in STATE:
        STATE[name] = torch.cuda.Stream(priority=prio)
    return STATE[name]


def qk(z, g):
    """Flash operands: q.k = 2g z.z' - g|z'|^2 (the per-row -g|z|^2 is restored from the log-sum-exp)."""
    zn = z.square().sum(1)
    one = torch.ones_like(zn)[:, None]
    q = F.pad(torch.cat([z * math.sqrt(2 * g), one], 1), (0, 3)).half()
    k = F.pad(torch.cat([z * math.sqrt(2 * g), (-g * zn)[:, None]], 1), (0, 3)).half()
    return q, k, zn


def kapply(q, k, qn, vh, g):
    """exp(2g q.k - g|q|^2 - g|k|^2) @ vh by the flash kernel (value width padded to the q/k width).

    Flash attention returns softmax(q k') v and the row log-sum-exp; multiplying by exp(lse) undoes the softmax
    normalization, and the extra -g|q|^2 completes the RBF exponent."""
    n = q.shape[0]
    v = F.pad(vh, (0, q.shape[1] - vh.shape[1]))
    r = torch.ops.aten._scaled_dot_product_flash_attention(q[None, None], k[None, None], v[None, None], scale=1.0)
    return r[0][0, 0, :, :vh.shape[1]].float() * torch.exp(r[1][0, 0, :n] - g * qn)[:, None]


def chol(a):
    """Lower Cholesky factor, without the host sync of torch.linalg.cholesky's error check."""
    return torch.linalg.cholesky_ex(a)[0]


def tri_inv(L):
    """L^-1 for lower-triangular L; column blocks (each needs only L[j0:, j0:]) on parallel streams."""
    m = L.shape[0]
    eye = torch.eye(m, device=L.device)
    out = torch.zeros(m, m, device=L.device)
    cur = torch.cuda.current_stream()
    bs = m // TPAR
    sts = []
    for p in range(TPAR):
        j0, j1 = p * bs, (m if p == TPAR - 1 else (p + 1) * bs)
        st = stream(f"tri{p}")
        st.wait_stream(cur)
        with torch.cuda.stream(st):
            out[j0:, j0:j1] = torch.linalg.solve_triangular(L[j0:, j0:], eye[j0:, j0:j1], upper=False)
        sts.append(st)
    for st in sts:
        cur.wait_stream(st)
    return out


def split16(r):
    """fp32 r as two fp16 halves [hi, lo] side by side (r = hi + lo to about fp32 accuracy), so an fp16 GEMM
    with fp32 output can multiply by r without rounding it to fp16."""
    hi = r.half()
    return torch.cat([hi, (r - hi.float()).half()], 1)


def preconditioner(z):
    """Nystrom preconditioner: r -> r - C16 Q^-1 C16' r with Q = mu (W + jitter I) + C16'C16 (any C keeps it
    positive definite as long as the same C16 is used in Q and in the application).

    C16 = K(z, landmarks) in fp16 and W = K(landmarks, landmarks), with the first RANK rows as landmarks and
    mu = PRE_RIDGE. By the Woodbury identity this is mu (C W^-1 C' + mu I)^-1, a scaled inverse of the Nystrom
    approximation of K + mu I; the scale does not change the CG iterates."""
    C16 = fexp(aug16(z, GAMMA, True), aug16(z[:RANK], GAMMA, False))
    W = kernel32(z[:RANK], z[:RANK], GAMMA)
    Q = torch.mm(C16.T, C16, out_dtype=torch.float32)
    Q.add_(W, alpha=PRE_RIDGE)
    Q.diagonal().add_(PRE_RIDGE * JITTER)
    li = tri_inv(chol(Q))
    lit = li.T.contiguous()

    def apply(r):
        """Q^-1 = L^-T L^-1 as two GEMMs; the C16 products use the hi/lo split, summing the two 10-column halves."""
        t = torch.mm(C16.T, split16(r), out_dtype=torch.float32)
        t = lit @ (li @ (t[:, :10] + t[:, 10:]))
        o = torch.mm(C16, split16(t), out_dtype=torch.float32)
        return r - o[:, :10] - o[:, 10:]
    return apply


def cg_plain(d, ap, nrm, w, r):
    """One unpreconditioned CG step per column (class); also returns the next direction in fp16."""
    a = nrm / (d * ap).sum(0)
    w = w + d * a
    r = r - ap * a
    nxt = (r * r).sum(0)
    d = r + d * (nxt / nrm)
    return w, r, d, d.half(), nxt


def pcg_a(d, ap, nrm, w, r):
    """First half of a preconditioned CG step: update the solution and the residual."""
    a = nrm / (d * ap).sum(0)
    return w + d * a, r - ap * a


def pcg_b(r, p, d, nrm):
    """Second half, given the preconditioned residual p: the next search direction (fp32 and fp16)."""
    nxt = (r * p).sum(0)
    d = p + d * (nxt / nrm)
    return d, d.half(), nxt


def last_comb(T, d, ap, nrm, beta: float):
    """Labels from the test scores of w4 + (1 + beta) a5 d4, given T = Kq [w4, d4] and the last matvec ap."""
    a = nrm / (d * ap).sum(0)
    return (T[:, :10] + T[:, 10:] * ((1 + beta) * a)).argmax(1)


def metric(xs, ys):
    """RFM metric step: returns L with M = L L', where M is the normalized AGOP of a kernel fit on (xs, ys).

    The gradient of f_c(x) = sum_j w_jc k(x, x_j) is 2g sum_j w_jc k(x, x_j) (x_j - x); the factor 2g drops out
    when M is scaled to a fixed trace."""
    ks = kernel32(xs, xs, GAMMA)
    ks.diagonal().add_(SUB_RIDGE)  # the diagonal cancels in the AGOP gradient, so ks serves both
    kmv = ks.half()
    w, r, d, nrm = torch.zeros_like(ys), ys.clone(), ys.clone(), ys.square().sum(0)
    dh = d.half()
    for _ in range(SUB_STEPS):
        w, r, d, dh, nrm = cg_plain(d, torch.mm(kmv, dh, out_dtype=torch.float32), nrm, w, r)
    fac, pts = ks[:SAMPLES], xs[:SAMPLES]
    scores = fac @ w
    wp = (w[:, :, None] * xs[:, None, :]).reshape(len(xs), -1)
    # TF32 only for this one large product (SAMPLES x SUB x 600); everything else stays in full fp32.
    torch.backends.cuda.matmul.allow_tf32 = True
    fw = fac @ wp
    torch.backends.cuda.matmul.allow_tf32 = False
    g = (fw.reshape(SAMPLES, 10, -1) - scores[:, :, None] * pts[:, None, :]).reshape(-1, xs.shape[1])
    m = g.T @ g
    m = m * ((1 - BLEND) * xs.shape[1] / m.trace())
    m.diagonal().add_(BLEND)
    return chol(m)


def fit_predict(train_x, train_y, test_x):
    """One full fit and prediction (recorded as a CUDA graph by classify)."""
    with torch.no_grad():
        x, q = train_x.float(), test_x.float()
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6) * x.shape[1] ** 0.5
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6) * q.shape[1] ** 0.5
        y = F.one_hot(train_y, 10).float()
        t = metric(x[:SUB], y[:SUB])
        z, zq = x @ t, q @ t
        main = torch.cuda.current_stream()
        side = stream("side", -1)
        side.wait_stream(main)
        with torch.cuda.stream(side):  # latency-bound preconditioner setup, overlapped with the flash operands
            prec = preconditioner(z)
        kq, kk, kn = qk(z, GAMMA)
        qq, _, qn = qk(zq, GAMMA)
        main.wait_stream(side)
        # PCG on (K + RIDGE I) w = y from w = 0; the ridge term uses the same fp16 direction as the flash matvec.
        w, r = torch.zeros_like(y), y.clone()
        d = prec(r)
        dh, nrm = d.half(), (r * d).sum(0)
        for _ in range(STEPS - 1):
            ap = kapply(kq, kk, kn, dh, GAMMA) + RIDGE * dh.float()
            w, r = pcg_a(d, ap, nrm, w, r)
            d, dh, nrm = pcg_b(r, prec(r), d, nrm)
        # Last step folded into the prediction: w5 is never formed.
        s3 = stream("pred")
        s3.wait_stream(main)
        with torch.cuda.stream(s3):    # test scores of [w4, d4] while the last train matvec runs
            T = kapply(qq, kk, qn, torch.cat([w.half(), dh], 1), GAMMA)
        ap = kapply(kq, kk, kn, dh, GAMMA) + RIDGE * dh.float()
        main.wait_stream(s3)
        return last_comb(T, d, ap, nrm, BETA)


def classify(train_x, train_y, test_x):
    """Predicted labels for test_x, from a kernel ridge fit on (train_x, train_y) made in this call."""
    key = ("g", tuple(train_x.shape), tuple(test_x.shape), train_x.dtype, test_x.dtype, train_y.dtype)
    if key not in STATE:  # warm-up call: run eagerly, then capture the whole fit on the caller's own tensors
        args = (train_x, train_y, test_x)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fit_predict(*args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fit_predict(*args)
        STATE[key] = (graph, args, out)
        import gc
        gc.collect()
        gc.freeze()   # keep the warm-up's objects out of every later collection
    graph, args, out = STATE[key]
    for buf, x in zip(args, (train_x, train_y, test_x)):
        if buf.data_ptr() != x.data_ptr():   # not the captured tensors: copy the inputs in
            buf.copy_(x)
    graph.replay()
    return out.clone()
