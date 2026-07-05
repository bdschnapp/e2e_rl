"""Batched matrix exponential + zero-order-hold discretisation (blocker #1).

The scalar env calls ``scipy.signal.cont2discrete(..., method="zoh")`` once PER
ENV PER STEP; scipy is CPU-only and un-batchable, so it is the single biggest
obstacle to a GPU-parallel env. This module replaces it with a batched matrix
exponential written purely in ``xp`` (numpy or cupy), so the ZOH for all N envs
is one vectorised call that runs on-device.

``expm_batch`` implements the scaling-and-squaring Padé-13 scheme (Higham 2005),
the same algorithm ``scipy.linalg.expm`` uses, so it matches scipy to ~1e-12 on
the 5x5 augmented matrices used here. A single global squaring count (the max over
the batch) keeps the squaring loop uniform — over-scaling any individual env is
still exact since expm(A) = (expm(A/2^s))^(2^s) for any s.

``zoh_discretize(A, B, dt)`` mirrors scipy's cont2discrete "zoh": exponentiate the
augmented block matrix [[A, B],[0, 0]]*dt and slice out (Ad, Bd).
"""

from __future__ import annotations

from .backend import xp

# Padé-13 numerator/denominator coefficients (Higham).
_B13 = (
    64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
    1187353796428800.0, 129060195264000.0, 10559470521600.0,
    670442572800.0, 33522128640.0, 1323241920.0,
    40840800.0, 960960.0, 16380.0, 182.0, 1.0,
)
_THETA13 = 5.371920351148152


def expm_batch(A):
    """Matrix exponential of a batch of square matrices.

    A: (N, k, k) array on the active backend. Returns (N, k, k).
    """
    A = xp.asarray(A, dtype=xp.float64)
    N, k, _ = A.shape
    ident = xp.eye(k, dtype=A.dtype)
    I = xp.broadcast_to(ident, (N, k, k))

    # 1-norm per matrix = max column abs-sum.
    norms = xp.max(xp.sum(xp.abs(A), axis=1), axis=1)  # (N,)
    max_norm = float(xp.max(norms)) if N else 0.0
    if max_norm > 0.0:
        import math
        s = max(0, int(math.ceil(math.log2(max_norm / _THETA13)))) if max_norm > _THETA13 else 0
    else:
        s = 0

    As = A / (2.0 ** s)

    b = _B13
    A2 = xp.matmul(As, As)
    A4 = xp.matmul(A2, A2)
    A6 = xp.matmul(A4, A2)

    U = xp.matmul(
        As,
        xp.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I,
    )
    V = (xp.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
         + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I)

    P = U + V
    Q = -U + V
    R = xp.linalg.solve(Q, P)

    for _ in range(s):
        R = xp.matmul(R, R)
    return R


def zoh_discretize(A, B, dt: float):
    """Batched zero-order-hold discretisation (== scipy cont2discrete 'zoh').

    A: (N, n, n), B: (N, n, m). Returns (Ad (N,n,n), Bd (N,n,m)).
    """
    A = xp.asarray(A, dtype=xp.float64)
    B = xp.asarray(B, dtype=xp.float64)
    N, n, _ = A.shape
    m = B.shape[2]
    M = xp.zeros((N, n + m, n + m), dtype=xp.float64)
    M[:, :n, :n] = A * dt
    M[:, :n, n:] = B * dt
    em = expm_batch(M)
    return em[:, :n, :n], em[:, :n, n:]
