# Derived from PR116 by @islamborghini (credited DeepSeek V4.1 Flash/OpenCode).
# Modification: one metric update, trace-normalized coupled Newton-Schulz square root, five iterations.
import torch
import triton
import triton.language as tl
torch.backends.cuda.matmul.allow_tf32 = False
GAMMA, RIDGE = (0.02, 0.05)
RANK = 256
KIND = 'rbf'
NORMALIZE = True
USE_TRITON = True
METRIC_UPDATES, METRIC_BLEND = (1, 0.05)
STEPS_PILOT, STEPS_FINAL = (16, 16)
PILOT_SIZE = 10000
METRIC_SAMPLES = 512
USE_FUSED_RBF = True
USE_FUSION = True
PRODUCT_BM, PRODUCT_BK, PRODUCT_SPLITS = (64, 64, 8)
CG_BN = 512
BLOCK_M, BLOCK_N, BLOCK_K = (64, 64, 64)
NUM_WARPS, NUM_STAGES = (4, 1)
GRAM_HALF = True
DIRECT_HALF_STORE = False
RADIAL_WEIGHT = 0.5
RADIAL_EPS = 1e-06
RADIAL_STANDARDIZE = True
NYSTROM_JITTER, HALF_JITTER = (0.0001, 0.001)
LOCAL_STRENGTH = 0.55
LOCAL_SELECT = 2048
LOCAL_NEIGHBORS = 64
LOCAL_RIDGE = 0.05
LOCAL_SCALE = 0
LOCAL_METRIC_POWER = 0.75
LOCAL_RAW = False
PSEUDO_COUNT = 7952
FUSE_LOCAL_COV = True

@triton.jit
def _product10(K, P, PART, N: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    cols = tl.arange(0, 16)
    acc = tl.zeros((BM, 16), tl.float32)
    for block in range(tl.cdiv(N, BK * SPLITS)):
        inner = (block * SPLITS + split) * BK + tl.arange(0, BK)
        a = tl.load(K + rows[:, None] * N + inner[None, :], (rows[:, None] < N) & (inner[None, :] < N), 0).to(tl.float32)
        b = tl.load(P + inner[:, None] * 10 + cols[None, :], (inner[:, None] < N) & (cols[None, :] < 10), 0)
        acc = tl.dot(a, b, acc, input_precision='tf32x3')
    tl.store(PART + split * N * 10 + rows[:, None] * 10 + cols[None, :], acc, (rows[:, None] < N) & (cols[None, :] < 10))

@triton.jit
def _product10_reduce(PART, P, OUT, N: tl.constexpr, RIDGE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    partial = tl.load(PART + splits[:, None] * N * 10 + index[None, :], index[None, :] < N * 10, 0)
    p = tl.load(P + index, index < N * 10, 0)
    tl.store(OUT + index, tl.sum(partial, 0) + RIDGE * p, index < N * 10)

@triton.jit
def _rbf_kernel(X, Z, OUT, NX, NZ, D, GAMMA, LAPLACIAN: tl.constexpr, HALF: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    xn = tl.zeros((BM,), tl.float32)
    zn = tl.zeros((BN,), tl.float32)
    for k0 in range(0, tl.cdiv(D, BK)):
        kk = k0 * BK + rk
        kmask = kk < D
        x = tl.load(X + rm[:, None] * D + kk[None, :], (rm[:, None] < NX) & kmask[None, :], 0.0)
        z = tl.load(Z + rn[:, None] * D + kk[None, :], (rn[:, None] < NZ) & kmask[None, :], 0.0)
        acc = tl.dot(x, tl.trans(z), acc, input_precision='tf32x3')
        xn += tl.sum(x * x, 1)
        zn += tl.sum(z * z, 1)
    distance = tl.maximum(xn[:, None] + zn[None, :] - 2.0 * acc, 0.0)
    if LAPLACIAN:
        distance = tl.sqrt(distance)
    k = tl.exp(-GAMMA * distance)
    if HALF:
        k = k.to(tl.float16)
    tl.store(OUT + rm[:, None] * NZ + rn[None, :], k, (rm[:, None] < NX) & (rn[None, :] < NZ))

@triton.jit
def _local_moments(QP, NBP, AP, WP, RP, K: tl.constexpr, D: tl.constexpr, KP: tl.constexpr, DP: tl.constexpr, LAM: tl.constexpr):
    pid = tl.program_id(0)
    rk = tl.arange(0, KP)
    rd = tl.arange(0, DP)
    km = rk < K
    dm = rd < D
    q = tl.load(QP + pid * D + rd, dm, 0.0)
    nb = tl.load(NBP + pid * K * D + rk[:, None] * D + rd[None, :], km[:, None] & dm[None, :], 0.0)
    mu = tl.sum(nb, 0) / K
    xc = tl.where(km[:, None], nb - mu[None, :], 0.0)
    delta = q - mu
    raw = tl.sum(tl.where(dm, delta * delta, 0.0))
    w = tl.sum(xc * delta[None, :], 1)
    a = tl.dot(xc, tl.trans(xc), input_precision='tf32x3') / K
    a = a + LAM * tl.where(rk[:, None] == rk[None, :], 1.0, 0.0)
    tl.store(AP + pid * K * K + rk[:, None] * K + rk[None, :], a, km[:, None] & km[None, :])
    tl.store(WP + pid * K + rk, w, km)
    tl.store(RP + pid, raw)

@triton.jit
def _dot_tile(X, Y, PART, N, C: tl.constexpr, CPAD: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    col = tl.arange(0, CPAD)
    cmask = col < C
    offs = rows[:, None] * C + col[None, :]
    m = (rows < N)[:, None] & cmask[None, :]
    x = tl.load(X + offs, m, 0.0)
    y = tl.load(Y + offs, m, 0.0)
    tl.store(PART + pid * CPAD + col, tl.sum(x * y, axis=0), mask=cmask)

@triton.jit
def _dot_sum(PART, OUT, T, C: tl.constexpr, CPAD: tl.constexpr, BLK: tl.constexpr):
    col = tl.arange(0, CPAD)
    cmask = col < C
    acc = tl.zeros((CPAD,), tl.float32)
    for t0 in range(0, T, BLK):
        idx = t0 + tl.arange(0, BLK)
        m = (idx < T)[:, None] & cmask[None, :]
        acc += tl.sum(tl.load(PART + idx[:, None] * CPAD + col[None, :], m, 0.0), axis=0)
    tl.store(OUT + col, acc, mask=cmask)

@triton.jit
def _axpy_update(W, R, D, A, NORM, DENOM, N, C: tl.constexpr, CPAD: tl.constexpr, BN: tl.constexpr):
    col = tl.arange(0, CPAD)
    cmask = col < C
    norm = tl.load(NORM + col, mask=cmask, other=0.0)
    denom = tl.load(DENOM + col, mask=cmask, other=1.0)
    alpha = norm / tl.maximum(denom, 1e-20)
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    offs = rows[:, None] * C + col[None, :]
    m = (rows < N)[:, None] & cmask[None, :]
    d = tl.load(D + offs, m, 0.0)
    a = tl.load(A + offs, m, 0.0)
    w = tl.load(W + offs, m, 0.0)
    r = tl.load(R + offs, m, 0.0)
    tl.store(W + offs, w + d * alpha[None, :], m)
    tl.store(R + offs, r - a * alpha[None, :], m)

@triton.jit
def _dir_update(DIR, PRE, NEXT, CUR, OUT, N, C: tl.constexpr, CPAD: tl.constexpr, BN: tl.constexpr):
    col = tl.arange(0, CPAD)
    cmask = col < C
    nxt = tl.load(NEXT + col, mask=cmask, other=0.0)
    cur = tl.load(CUR + col, mask=cmask, other=1.0)
    factor = nxt / tl.maximum(cur, 1e-20)
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    offs = rows[:, None] * C + col[None, :]
    m = (rows < N)[:, None] & cmask[None, :]
    pre = tl.load(PRE + offs, m, 0.0)
    d = tl.load(DIR + offs, m, 0.0)
    tl.store(DIR + offs, pre + d * factor[None, :], m)
    if pid == 0:
        tl.store(OUT + col, nxt, mask=cmask)

def make_workspace(n, c, device):
    splits = int(PRODUCT_SPLITS)
    return {'partial': torch.empty((splits, n, 10), device=device, dtype=torch.float32), 'out': torch.empty((n, c), device=device, dtype=torch.float32), 'dot_part': torch.empty((triton.cdiv(n, CG_BN), 16), device=device), 'denom': torch.empty(c, device=device), 'next': torch.empty(c, device=device), 'norm_a': torch.empty(c, device=device), 'norm_b': torch.empty(c, device=device)}

def _dot(x, y, out, workspace, n, c):
    tiles = triton.cdiv(n, CG_BN)
    _dot_tile[tiles,](x, y, workspace['dot_part'], n, c, 16, CG_BN)
    _dot_sum[1,](workspace['dot_part'], out, tiles, c, 16, 32)

def product(k, direction, ridge, workspace):
    if not USE_TRITON:
        return k.to(direction.dtype) @ direction + ridge * direction
    n = k.shape[0]
    partial, out = (workspace['partial'], workspace['out'][:n])
    splits, bm, bk = (int(PRODUCT_SPLITS), int(PRODUCT_BM), int(PRODUCT_BK))
    _product10[triton.cdiv(n, bm), splits](k, direction, partial, n, splits, bm, bk)
    _product10_reduce[triton.cdiv(n * 10, 256),](partial, direction, out, n, ridge, splits, 256)
    return out

def _fused_kernel(x, z, gamma, kind):
    train = GRAM_HALF and x is z
    direct = train and DIRECT_HALF_STORE
    x, z = (x.contiguous(), z.contiguous())
    nx, nz = (x.shape[0], z.shape[0])
    d = x.shape[1]
    out = torch.empty((nx, nz), device=x.device, dtype=torch.float16 if direct else torch.float32)
    grid = (triton.cdiv(nx, BLOCK_M), triton.cdiv(nz, BLOCK_N))
    _rbf_kernel[grid](x, z, out, nx, nz, d, gamma, kind == 'laplacian', direct, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out.half() if train and (not direct) else out

def kernel(x, z, gamma, kind):
    if USE_FUSED_RBF:
        return _fused_kernel(x, z, gamma, kind)
    distance = (x.square().sum(1)[:, None] + z.square().sum(1)[None, :] - 2 * (x @ z.T)).clamp_min_(0)
    if kind == 'laplacian':
        distance.sqrt_()
    return distance.mul_(-gamma).exp_()

def _nystrom_precond(k, ridge):
    m = min(RANK, len(k))
    c = k[:, :m].float().contiguous()
    landmark = c[:m, :m].clone()
    landmark.diagonal().add_(HALF_JITTER if k.dtype == torch.float16 else NYSTROM_JITTER)
    factor = torch.linalg.cholesky(landmark)
    u = torch.linalg.solve_triangular(factor, c.T.contiguous(), upper=False).T.contiguous()
    small = u.T @ u
    small.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(small)
    v = torch.linalg.solve_triangular(factor, u.T.contiguous(), upper=False).T.contiguous()

    def precondition(residual):
        return (residual - v @ (v.T @ residual)) / ridge
    return precondition

def _base_solve(k, targets, ridge, steps, precond, scratch=None):
    weights = torch.zeros_like(targets)
    residual = targets.clone()
    direction = precond(residual).clone()
    norm = (residual * direction).sum(0)
    for _ in range(steps):
        applied = product(k, direction, ridge, scratch)
        alpha = norm / direction.mul(applied).sum(0).clamp_min(1e-20)
        weights.add_(direction * alpha)
        residual.sub_(applied * alpha)
        preconditioned = precond(residual)
        next_norm = (residual * preconditioned).sum(0)
        direction = preconditioned + direction * (next_norm / norm.clamp_min(1e-20))
        norm = next_norm
    return weights

def _fused_solve_triton(k, targets, ridge, steps, precond, workspace):
    n, c = (k.shape[0], targets.shape[1])
    weights = torch.zeros_like(targets)
    residual = targets.clone()
    direction = precond(residual).clone()
    denom, nxt = (workspace['denom'], workspace['next'])
    cur, out = (workspace['norm_a'], workspace['norm_b'])
    grid = (triton.cdiv(n, CG_BN),)
    _dot(residual, direction, cur, workspace, n, c)
    for _ in range(steps):
        applied = product(k, direction, ridge, workspace)
        _dot(direction, applied, denom, workspace, n, c)
        _axpy_update[grid](weights, residual, direction, applied, cur, denom, n, c, 16, CG_BN)
        preconditioned = precond(residual)
        _dot(residual, preconditioned, nxt, workspace, n, c)
        _dir_update[grid](direction, preconditioned, nxt, cur, out, n, c, 16, CG_BN)
        cur, out = (out, cur)
    return weights

def solve(k, targets, ridge, steps, workspace):
    precond = _nystrom_precond(k, ridge) if RANK else None

    def precondition(residual):
        return precond(residual) if precond is not None else residual
    if USE_FUSION and USE_TRITON:
        return _fused_solve_triton(k, targets, ridge, steps, precondition, workspace)
    return _base_solve(k, targets, ridge, steps, precondition, workspace)

def prepare(train_x, test_x):
    x, q = (train_x.float(), test_x.float())
    if not NORMALIZE:
        return (x, q)
    radius_x = x.norm(dim=1)
    radius_q = q.norm(dim=1)
    scale = x.shape[1] ** 0.5
    x = x / radius_x.clamp_min(RADIAL_EPS).unsqueeze(1) * scale
    q = q / radius_q.clamp_min(RADIAL_EPS).unsqueeze(1) * scale
    if RADIAL_WEIGHT == 0:
        return (x, q)
    log_x = torch.log(radius_x.clamp_min(RADIAL_EPS))
    log_q = torch.log(radius_q.clamp_min(RADIAL_EPS))
    if RADIAL_STANDARDIZE:
        mean = log_x.mean()
        std = log_x.std(unbiased=False).clamp_min(RADIAL_EPS)
        log_x = (log_x - mean) / std
        log_q = (log_q - mean) / std
    x = torch.cat([x, (RADIAL_WEIGHT * log_x).unsqueeze(1)], dim=1)
    q = torch.cat([q, (RADIAL_WEIGHT * log_q).unsqueeze(1)], dim=1)
    return (x, q)

def gradient_metric(x, transform, weights, gamma, kind, samples, blend):
    z = x @ transform
    points = z[:min(samples, len(z))]
    factor = kernel(points, z, gamma, kind)
    if kind == 'laplacian':
        distance = (points.square().sum(1)[:, None] + z.square().sum(1)[None] - 2 * (points @ z.T)).clamp_min_(0)
        factor = factor / distance.sqrt_().clamp_min_(1e-06)
        index = torch.arange(len(points), device=x.device)
        factor[index, index] = 0
    scores = factor @ weights
    weighted_points = (weights[:, :, None] * z[:, None, :]).reshape(len(z), -1)
    gradients = (factor @ weighted_points).reshape(len(points), 10, -1) - scores[:, :, None] * points[:, None, :]
    gradients = gradients @ transform.T
    gradients = gradients.reshape(-1, x.shape[1])
    metric = gradients.T @ gradients
    metric = metric * (x.shape[1] / metric.trace().clamp_min(1e-12))
    metric = (1 - blend) * metric + blend * torch.eye(x.shape[1], device=x.device)
    alpha = torch.linalg.matrix_norm(metric).clamp_min(1e-12)
    Y = metric / alpha
    I = torch.eye(len(metric), device=metric.device, dtype=metric.dtype)
    Z = I.clone()
    for _ in range(5):
        B = 0.5 * (3 * I - Z @ Y)
        Y = Y @ B
        Z = B @ Z
    T = Y * alpha.sqrt()
    return T * (T.shape[0] / T.square().sum().clamp_min(1e-12)).sqrt()

def _solve_local(a, b):
    factor = torch.linalg.cholesky_ex(a, check_errors=False)[0]
    return torch.cholesky_solve(b, factor)

def _fused_moments(q, nb, k, lam):
    Q, D = q.shape
    KP = max(16, triton.next_power_of_2(k))
    DP = max(32, triton.next_power_of_2(D))
    a = torch.empty((Q, k, k), device=q.device, dtype=torch.float32)
    w = torch.empty((Q, k), device=q.device, dtype=torch.float32)
    raw = torch.empty(Q, device=q.device, dtype=torch.float32)
    _local_moments[Q,](q, nb, a, w, raw, k, D, KP, DP, float(lam))
    return (a, w, raw)

def _local_val(q, zc, k, lam, scale):
    dist = q.square().sum(1)[:, None] + zc.square().sum(1)[None] - 2 * (q @ zc.T)
    nb = zc[dist.topk(k, 1, largest=False).indices]
    if FUSE_LOCAL_COV and USE_TRITON and q.is_cuda and (k >= 2):
        a, w, raw = _fused_moments(q.contiguous(), nb.contiguous(), k, lam)
    else:
        mu = nb.mean(1, keepdim=True)
        xc = nb - mu
        delta = q - mu[:, 0, :]
        raw = delta.square().sum(1)
        w = (xc * delta[:, None, :]).sum(2)
        a = xc @ xc.transpose(1, 2) / k + lam * torch.eye(k, dtype=zc.dtype, device=zc.device)
    y = _solve_local(a, w.unsqueeze(2)).squeeze(2)
    val = (raw - (w * y).sum(1) / k) * (-0.5 / lam)
    if scale:
        val = val - 0.5 * torch.linalg.slogdet(a)[1]
    return val

def pseudo_split(scores):
    nq = scores.shape[0]
    m = min(LOCAL_SELECT, nq)
    top = scores.topk(2, 1).values
    order = torch.argsort(top[:, 0] - top[:, 1])
    k = min(int(PSEUDO_COUNT), max(nq - m, 0))
    return (order[:m], order[nq - k:] if k else order[:0])

def local_scores(z, qz, scores, train_y, k):
    pick, src = pseudo_split(scores)
    q, supp = (qz[pick], qz[src])
    labels = scores.argmax(1)[src]
    out = q.new_zeros((pick.shape[0], 10))
    for c in range(10):
        zc = z[train_y == c]
        if PSEUDO_COUNT and src.shape[0]:
            zc = torch.cat([zc, supp[labels == c]], 0)
        if zc.shape[0] >= 2:
            out[:, c] = _local_val(q, zc, min(int(k), zc.shape[0]), float(LOCAL_RIDGE), int(LOCAL_SCALE))
    out = out - out.mean(1, keepdim=True)
    return (pick, out / out.std(1, keepdim=True).clamp_min(1e-06))

def _powered(z, transform, power):
    w = transform.square().sum(0).clamp_min(1e-06) ** ((float(power) - 1.0) / 2.0)
    return z * w

def fit_scores(train_x, train_y, test_x):
    with torch.no_grad():
        x, q = prepare(train_x, test_x)
        targets = torch.zeros((len(x), 10), device=x.device)
        targets.scatter_(1, train_y[:, None], 1.0)
        workspace = make_workspace(len(x), 10, x.device)
        size = min(PILOT_SIZE, len(x))
        pilot_x, pilot_targets = (x[:size], targets[:size])
        transform = torch.eye(x.shape[1], device=x.device)
        for _ in range(METRIC_UPDATES):
            z = pilot_x @ transform
            k = kernel(z, z, GAMMA, KIND)
            k.diagonal().fill_(1.0)
            weights = solve(k, pilot_targets, RIDGE, STEPS_PILOT, workspace)
            transform = gradient_metric(pilot_x, transform, weights, GAMMA, KIND, METRIC_SAMPLES, METRIC_BLEND)
        z = x @ transform if METRIC_UPDATES else x
        k = kernel(z, z, GAMMA, KIND)
        k.diagonal().fill_(1.0)
        weights = solve(k, targets, RIDGE, STEPS_FINAL, workspace)
        qz = q @ transform if METRIC_UPDATES else q
        scores = torch.cat([kernel(chunk, z, GAMMA, KIND) @ weights for chunk in qz.split(2000)])
        if LOCAL_STRENGTH:
            if LOCAL_RAW:
                lz, lq = (train_x.float(), test_x.float())
            else:
                lz = _powered(z, transform, LOCAL_METRIC_POWER)
                lq = _powered(qz, transform, LOCAL_METRIC_POWER)
            pick, local = local_scores(lz, lq, scores, train_y, LOCAL_NEIGHBORS)
            scores[pick] = scores[pick] + LOCAL_STRENGTH * local
        return scores

def classify(train_x, train_y, test_x):
    return fit_scores(train_x, train_y, test_x).argmax(1)
