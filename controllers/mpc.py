import numpy as np
from scipy.linalg import expm
import scipy.sparse as sparse
import osqp


def _as_col(X):
    """Convert input to a 2D numpy array with a single column."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _discretize_zoh(A, B, Ts):
    """
    Discretize xdot = A x + B u with zero-order hold over sample time Ts.
    Returns (Ad, Bd).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    m = B.shape[1]

    # Augmented matrix exponential:
    # exp( [A B; 0 0] * Ts ) = [Ad Bd; 0 I]
    M = np.zeros((n + m, n + m), dtype=float)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * Ts)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def calculate_curvature(x_p, y_p, path_x, path_y):
    """
    Signed curvature kappa of path at the nearest sample to (x_p, y_p).
    """
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    n = path_x.size
    if n < 3:
        return 0.0

    # Nearest index
    distances = np.hypot(path_x - x_p, path_y - y_p)
    idx = int(np.argmin(distances))

    # Finite differences
    if idx == 0:
        dx  = path_x[idx + 1] - path_x[idx]
        dy  = path_y[idx + 1] - path_y[idx]
        d2x = path_x[idx + 2] - 2.0 * path_x[idx + 1] + path_x[idx]
        d2y = path_y[idx + 2] - 2.0 * path_y[idx + 1] + path_y[idx]
    elif idx == n - 1:
        dx  = path_x[idx] - path_x[idx - 1]
        dy  = path_y[idx] - path_y[idx - 1]
        d2x = path_x[idx] - 2.0 * path_x[idx - 1] + path_x[idx - 2]
        d2y = path_y[idx] - 2.0 * path_y[idx - 1] + path_y[idx - 2]
    else:
        dx  = 0.5 * (path_x[idx + 1] - path_x[idx - 1])
        dy  = 0.5 * (path_y[idx + 1] - path_y[idx - 1])
        d2x = path_x[idx + 1] - 2.0 * path_x[idx] + path_x[idx - 1]
        d2y = path_y[idx + 1] - 2.0 * path_y[idx] + path_y[idx - 1]

    denom = (dx*dx + dy*dy)**1.5
    if denom < 1e-12:
        return 0.0
    # SIGNED curvature (right-hand rule)
    kappa = (dx * d2y - dy * d2x) / denom
    return float(kappa)


def wrap_to_pi(x):
    """Wrap angle(s) to (-pi, pi]."""
    x = np.asarray(x, dtype=float)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class TractorTrailerSteeringMPC:
    def __init__(self, args=None):
        self.L1 = 4.0  # Tractor length
        self.L2 = 10.0  # Trailer length
        self.L2C = np.finfo(float).eps  # 2.220446049250313e-16  (not sure why MATLAB code uses this)

        self.nx = 6  # Number of states (X, Y, psi1, psi2, theta, e)
        self.nu = 1  # Number of inputs
        self.Ts = 0.1  # Sampling time

        self.Q = np.diag([0.0, 2000.0, 2000.0, 500.0])  # State cost
        self.R = np.diag([1.0])  # Input cost
        self.P = np.diag([1e5])  # Input-delta cost
        self.N = 40  # Horizon length

        self.umin = -np.pi / 6  # Min steering angle
        self.umax = np.pi / 6   # Max steering angle
        self.delta_umin = -360 * self.Ts / (24*180/np.pi)
        self.delta_umax = 360 * self.Ts / (24*180/np.pi)

        self.v = []  # Store last solution
        self.n_consecutive_failures = 0

    def reset(self):
        """Clear warm-start buffer and failure counter between episodes."""
        self.v = []
        self.n_consecutive_failures = 0

    def A_Matrix(self, delta_0, psi1_0, psi2_0, vx):
        """
        Python/Numpy translation of the MATLAB function A_Matrix.
        Inputs can be floats (typical) or NumPy scalars.

        Parameters
        ----------
        delta_0, psi1_0, psi2_0 : float
        vx : float

        Returns
        -------
        A : (4, 4) numpy.ndarray
        """
        t2 = np.cos(delta_0)
        t3 = np.cos(psi2_0)
        t4 = np.sin(delta_0)
        t5 = np.sin(psi2_0)
        t6 = self.L2 * t2 * t3
        t7 = self.L2 * t2 * t5
        t8 = self.L2C * t3 * t4
        t9 = self.L2C * t4 * t5
        t10 = self.L1 * t6
        t11 = self.L1 * t7
        t12 = self.L1 * t8
        t13 = self.L1 * t9
        # t14 = -t9; t15 = -t13
        t16 = t6 - t9  # t6 + t14
        t17 = t11 + t12
        t18 = t10 - t13  # t10 + t15
        t19 = 1.0 / t18
        t20 = t19 ** 2
        t21 = t16 * t19 * vx

        term15 = t21 + t17 * t20 * vx * (t7 + t8)
        term16 = -t21 - t17 * t20 * vx * (t7 + t8 - self.L1 * (t3 ** 2) * t4 - self.L1 * t4 * (t5 ** 2))

        A = np.array([
            [0.0, 0.0, -vx * np.sin(psi1_0), 0.0],
            [0.0, 0.0, vx * np.cos(psi1_0), 0.0],
            [0.0, 0.0, 0.0, term15],
            [0.0, 0.0, 0.0, term16],
        ], dtype=float)

        return A

    def B_Matrix(self, delta_0, psi2_0, vx):
        """
        Python/Numpy translation of the MATLAB function B_Matrix.

        Parameters
        ----------
        delta_0, psi2_0 : float
        vx : float

        Returns
        -------
        B : (4, 1) numpy.ndarray   # column vector to mirror MATLAB
        """
        t2 = np.cos(delta_0)
        t3 = np.cos(psi2_0)
        t4 = np.sin(delta_0)
        t5 = np.sin(psi2_0)
        t6 = t3 ** 2
        t7 = t5 ** 2
        t8 = self.L2C * t2 * t3
        t9 = self.L2 * t2 * t5
        t10 = self.L2C * t3 * t4
        t11 = self.L2 * t4 * t5
        t12 = self.L1 * self.L2 * t2 * t3
        t13 = self.L1 * self.L2 * t3 * t4
        t14 = self.L1 * self.L2C * t2 * t5
        t15 = self.L1 * self.L2C * t4 * t5
        # t16 = -t15
        t17 = t13 + t14
        t18 = t12 - t15  # t12 + (-t15)
        t19 = 1.0 / t18
        t20 = t19 ** 2

        b3 = t19 * vx * (t8 - t11) + t17 * t20 * vx * (t9 + t10)
        b4 = t19 * vx * (-t8 + t11 + self.L1 * t2 * t6 + self.L1 * t2 * t7) \
             - t17 * t20 * vx * (t9 + t10 - self.L1 * t4 * t6 - self.L1 * t4 * t7)

        B = np.array([[0.0],
                      [0.0],
                      [b3],
                      [b4]], dtype=float)
        return B

    def W_Matrix(self, delta_0, psi1_0, psi2_0, vx):
        """
        Python/Numpy translation of the MATLAB function W_Matrix.

        Parameters
        ----------
        delta_0, psi1_0, psi2_0 : float
        vx : float

        Returns
        -------
        W : (4, 1) numpy.ndarray   # column vector to mirror MATLAB
        """
        t2 = np.cos(delta_0)
        t3 = np.cos(psi1_0)
        t4 = np.cos(psi2_0)
        t5 = np.sin(delta_0)
        t6 = np.sin(psi1_0)
        t7 = np.sin(psi2_0)
        t8 = t4 ** 2
        t9 = t7 ** 2
        t10 = self.L2 * t2 * t4
        t11 = self.L2C * t2 * t4
        t12 = self.L2 * t2 * t7
        t13 = self.L2C * t4 * t5
        t14 = self.L2 * t5 * t7
        t15 = self.L2C * t5 * t7
        t18 = self.L1 * self.L2 * t4 * t5
        t19 = self.L1 * self.L2C * t2 * t7
        t16 = self.L1 * t10
        t17 = self.L1 * t12
        t20 = self.L1 * t13
        t21 = self.L1 * t15
        t22 = -t15
        t23 = self.L1 * t5 * t8
        t24 = self.L1 * t5 * t9
        t28 = t12 + t13
        t31 = t18 + t19
        t25 = -t21
        t26 = -t23
        t27 = -t24
        t29 = t10 + t22
        t30 = t17 + t20
        t32 = t16 + t25
        t35 = t26 + t27 + t28
        t33 = 1.0 / t32
        t34 = t33 ** 2
        t36 = t29 * t33 * vx

        w1 = t3 * vx + psi1_0 * t6 * vx
        w2 = t6 * vx - psi1_0 * t3 * vx
        w3 = (-delta_0 * (t33 * vx * (t11 - t14) + t28 * t31 * t34 * vx)
              - psi2_0 * (t36 + t28 * t30 * t34 * vx)
              + t28 * t33 * vx)
        w4 = (-delta_0 * (t33 * vx * (-t11 + t14 + self.L1 * t2 * t8 + self.L1 * t2 * t9)
                          - t31 * t34 * t35 * vx)
              + psi2_0 * (t36 + t30 * t34 * t35 * vx)
              - t33 * t35 * vx)

        W = np.array([[w1],
                      [w2],
                      [w3],
                      [w4]], dtype=float)
        return W

    def delta_ff(self, kappa, psi2):
        """
            Python/Numpy translation of the MATLAB function delta_ff.
            Works with scalars or array-like psi2 (broadcasting). Returns angles (radians).

            Returns
            -------
            delta_ff : np.ndarray
                Shape (2, 1) if inputs are scalars; (2, N) if psi2 is length-N.
            """
        # Core terms
        t2 = self.L1 * kappa
        t3 = self.L2 * kappa
        t4 = self.L2C * kappa
        t5 = -t4

        # Complex exponentials (use 1j for sqrt(-1))
        t7 = np.exp(1j * psi2)  # exp(i*psi2)
        t10 = np.exp(0.5j * psi2)  # exp(i*psi2/2)
        t11 = np.exp(1.5j * psi2)  # exp(3i*psi2/2)

        t12 = t2 * t7
        t13 = t3 * t7
        t14 = t4 * t7
        t15 = 1j * t7
        t16 = t3 * t10
        t17 = t2 * t11
        t18 = t5 * t7
        t19 = t4 * t10
        t20 = 1j * t11
        t21 = t5 * t10

        t22 = t2 + t13 + t18 - 1j
        t23 = t3 + t5 + t12 + t15
        t24 = t16 + t17 + t20 + t21

        t25 = 1.0 / t24
        t26 = t22 * t23
        t27 = -t26
        t28 = np.sqrt(t27)  # complex sqrt (principal branch)
        t29 = t25 * t28

        # Two possible angles
        phi1 = np.angle(t29)
        phi2 = np.angle(-t29)

        # Ensure 2×N (or 2×1) output like MATLAB column stacking
        phi1 = np.atleast_1d(phi1)
        phi2 = np.atleast_1d(phi2)
        return np.vstack([phi1, phi2])

    def solve_psi2(self, kappa):
        """
        Python translation of the MATLAB solve_psi2.

        Solves for psi2 in: sin(psi2) - L1*kappa*cos(psi2) + L2C*kappa = 0

        Parameters
        ----------
        kappa : float or array-like

        Returns
        -------
        psi2 : np.ndarray
            Shape (2,) for scalar kappa, or (2, N) for vector kappa.
            First row is psi2_1, second row is psi2_2 (both wrapped to (-pi, pi]).
        """
        kappa = np.asarray(kappa, dtype=float)

        R = np.sqrt(1.0 + (self.L1 * kappa) ** 2)  # combined amplitude
        alpha = np.arctan2(-self.L1 * kappa, 1.0)  # phase shift
        rhs = -self.L2C * kappa / R  # RHS argument to asin

        # numerical safety
        rhs = np.clip(rhs, -1.0, 1.0)

        asin_rhs = np.arcsin(rhs)
        psi2_1 = asin_rhs - alpha
        psi2_2 = np.pi - asin_rhs - alpha

        psi2_1 = wrap_to_pi(psi2_1)
        psi2_2 = wrap_to_pi(psi2_2)

        psi2 = np.vstack([psi2_1, psi2_2])
        return psi2

    def linearize(self, delta0, psi10, psi20, vx):
        """
        Linearize and discretize the tractor-trailer model at the given operating point.

        Parameters:
        delta0 : float
            Steering angle of the tractor (radians)
        psi10 : float
            Yaw angle of the trailer (radians)
        psi20 : float
            Hitch angle (radians)
        vx : float
            Longitudinal velocity of the tractor (m/s)
        """
        A = self.A_Matrix(delta0, psi10, psi20, vx)             # (nx, nx)
        B = _as_col(self.B_Matrix(delta0, psi20, vx))           # (nx, nu)
        W = _as_col(self.W_Matrix(delta0, psi10, psi20, vx))    # (nx, nw)

        BW = np.hstack([B, W])                                  # (nx, nu+nw)

        A_d, BW_d = _discretize_zoh(A, BW, self.Ts)
        B_d = BW_d[:, :self.nu]                                 # (nx, nu)
        W_d = BW_d[:, self.nu:]                                 # (nx, nw)
        return A_d, B_d, W_d

    def system_metrics(self, A, B, d, U_0, X_0, X_ref):
        """
        Python/Numpy translation of the MATLAB function system_metrics.

        Parameters
        ----------
        A : (n_x, n_x) array_like
        B : (n_x, n_u) array_like
        d : (n_x,) or (n_x,1) array_like               # affine term per step
        U_0 : (n_u,) or (n_u,1) array_like             # previous input
        X_0 : (n_x,) or (n_x,1) array_like             # current state
        X_ref : (N*n_x,) or (N*n_x,1) array_like       # stacked reference over horizon

        Returns
        -------
        H : (N*n_u, N*n_u) ndarray
        f : (N*n_u, 1) ndarray
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        Q = np.asarray(self.Q, dtype=float)
        R = np.asarray(self.R, dtype=float)
        P = np.asarray(self.P, dtype=float)

        n_x = A.shape[0]
        n_u = B.shape[1]

        # columnize vectors
        def col(v):
            v = np.asarray(v, dtype=float)
            return v.reshape(-1, 1)

        d = col(d)
        U_0 = col(U_0)
        X_0 = col(X_0)
        X_ref = col(X_ref)

        # Precompute powers A^k for k=0..N
        A_powers = [np.eye(n_x, dtype=float)]
        for _ in range(1, self.N + 1):
            A_powers.append(A_powers[-1] @ A)

        # ---- Ax (stacked A^i) ----
        Ax = np.zeros((self.N * n_x, n_x), dtype=float)
        for i in range(1, self.N + 1):
            Ax[(i - 1) * n_x: i * n_x, :] = A_powers[i]

        # ---- Bu (block-lower-triangular with A^(i-j) B) ----
        Bu = np.zeros((self.N * n_x, self.N * n_u), dtype=float)
        for i in range(1, self.N + 1):
            for j in range(1, i + 1):
                block = A_powers[i - j] @ B
                r0, r1 = (i - 1) * n_x, i * n_x
                c0, c1 = (j - 1) * n_u, j * n_u
                Bu[r0:r1, c0:c1] = block

        # ---- D (stacked affine sums) ----
        D = np.zeros((self.N * n_x, 1), dtype=float)
        current_sum = np.zeros((n_x, 1), dtype=float)
        for i in range(1, self.N + 1):
            current_sum = current_sum + (A_powers[i - 1] @ d)
            D[(i - 1) * n_x: i * n_x, :] = current_sum

        # ---- T (block first-difference operator on inputs) ----
        I_Nu = np.eye(self.N * n_u, dtype=float)
        subdiag = np.eye(self.N * n_u, k=-n_u, dtype=float)  # ones on block-subdiagonal
        T = I_Nu - subdiag

        # ---- t0 vector ----
        t0 = np.zeros((self.N * n_u, 1), dtype=float)
        t0[:n_u, :] = U_0

        # ---- Block-diagonal weights ----
        Qbar = np.kron(np.eye(self.N, dtype=float), Q)  # (N*n_x, N*n_x)
        Rbar = np.kron(np.eye(self.N, dtype=float), R)  # (N*n_u, N*n_u)
        Pbar = np.kron(np.eye(self.N, dtype=float), P)  # (N*n_u, N*n_u)

        # ---- Quadratic cost matrices ----
        H = 2.0 * (Bu.T @ Qbar @ Bu + Rbar + T.T @ Pbar @ T)
        H = 0.5 * (H + H.T)  # enforce symmetry

        f = 2.0 * (Bu.T @ Qbar @ (Ax @ X_0 + D - X_ref) - T.T @ Pbar @ t0)

        return H, f

    def system_constraints(self, A, B, U_prev_total, delta_ff):
        """
        Constraints on the *optimization variable* v = u_total - delta_ff.

        If your delta limits represent a rate in rad/s, multiply by Ts
        before building lb_delta_u / ub_delta_u.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        n_x = A.shape[0]
        n_u = B.shape[1]

        def col(v): return np.asarray(v, dtype=float).reshape(-1, 1)

        U_prev_total = col(U_prev_total)

        # Box constraints on total steering (per input)
        umin = float(self.umin)
        umax = float(self.umax)

        # Δ bounds are per-step increments (change this if you store rad/s)
        delta_umin = float(self.delta_umin)  # already per-step
        delta_umax = float(self.delta_umax)

        # T operator
        I = np.eye(self.N * n_u, dtype=float)
        sub = np.eye(self.N * n_u, k=-n_u, dtype=float)
        Aineq = I - sub  # (N*n_u, N*n_u); rows=constraints, cols=vars

        # t0 term uses previous TOTAL input
        t0 = np.zeros((self.N * n_u, 1), dtype=float)
        t0[:n_u, :] = U_prev_total

        # effect of constant delta_ff on Δu_total: [delta_ff, 0, 0, ...]^T
        tdff = np.zeros_like(t0)
        tdff[:n_u, :] = delta_ff  # scalar or (n_u,1)

        # replicate per-horizon boxes for v = u_total - delta_ff
        lb_u = np.full((self.N * n_u, 1), umin - float(delta_ff), dtype=float)
        ub_u = np.full((self.N * n_u, 1), umax - float(delta_ff), dtype=float)

        # rate bounds on Δu_total, shifted for delta_ff in the first block
        lb_delta_u = np.full((self.N * n_u, 1), delta_umin, dtype=float)
        ub_delta_u = np.full((self.N * n_u, 1), delta_umax, dtype=float)

        lbA = lb_delta_u + t0 - tdff
        ubA = ub_delta_u + t0 - tdff

        return Aineq, lbA, ubA, lb_u, ub_u

    def optimization(self, A, B, d_eff, U_prev_total, X_0, X_ref, delta_ff,
                     nWSR_max: int = 200, eps_reg: float = 1e-9,
                     print_level: str = "NONE"):
        """
        Solve for v = u_total - delta_ff using OSQP.

        OSQP solves: minimize 0.5 x^T P x + q^T x  s.t.  l <= A_c x <= u
        We encode:
          - Rate constraints:      lbA <= Aineq v <= ubA
          - Box constraints on v:  lb <= v <= ub    (add as extra rows: I v in [lb, ub])
        """
        H, f = self.system_metrics(A, B, d_eff, U_prev_total, X_0, X_ref)
        Aineq, lbA, ubA, lb_u, ub_u = self.system_constraints(A, B, U_prev_total, delta_ff)

        nV = H.shape[0]  # = N * n_u

        # Regularize H on the diagonal to ensure positive semidefiniteness
        if eps_reg and eps_reg > 0.0:
            H = H.copy()
            H.flat[::nV + 1] += eps_reg

        # Stack constraints into OSQP canonical form
        #   1) Rate constraints:      lbA <= Aineq v <= ubA
        #   2) Variable bounds (box): lb   <=    I v <= ub
        I = np.eye(nV, dtype=float)

        A_c = np.vstack([Aineq, I])            # shape (nC + nV, nV)
        l   = np.vstack([lbA.reshape(-1,1), lb_u.reshape(-1,1)]).ravel()
        u   = np.vstack([ubA.reshape(-1,1), ub_u.reshape(-1,1)]).ravel()

        # Convert to sparse (OSQP expects CSC)
        P = sparse.csc_matrix(H)
        q = np.ascontiguousarray(f.ravel(), dtype=float)
        A_c = sparse.csc_matrix(A_c)

        # Configure OSQP
        # Note: 'max_iter' is OSQP's iteration cap (analogous to nWSR_max spirit)
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A_c, l=l, u=u,
                   verbose=(print_level.upper() != "NONE"),
                   polish=True,
                   max_iter=int(max(1000, 5 * nWSR_max)),   # give it some headroom
                   eps_abs=1e-5, eps_rel=1e-5,
                   adaptive_rho=True)

        res = prob.solve()

        if res.info.status_val not in (1, 2):  # 1: solved, 2: solved inaccurate
            raise RuntimeError(f"OSQP failed: {res.info.status} (code {res.info.status_val})")

        v = np.asarray(res.x, dtype=float)
        return v

    def solve(self, trajectory, state):
        """
        trajectory: (N+1,4) rows [X, Y, psi1, psi2], row 0 is current measured state.
        state: (vx, U_prev_total)
        """
        traj = np.asarray(trajectory, dtype=float)
        assert traj.ndim == 2 and traj.shape[1] >= 4, "trajectory must be (M,4+)"

        nx = 4
        X_0, Y_0, psi1_0, psi2_0 = traj[0, :4]
        vx, U_prev_total = float(state[0]), float(state[1])

        # Linearize & discretize at the current operating point (around current total steer)
        A_d, B_d, W_d = self.linearize(U_prev_total, psi1_0, psi2_0, vx)

        # ---- Geometry feed-forward from path ----
        path_x = traj[:, 0]
        path_y = traj[:, 1]
        kappa = calculate_curvature(X_0, Y_0, path_x, path_y)

        psi2_candidates = self.solve_psi2(kappa)
        psi2_des = float(psi2_candidates[0]) if abs(psi2_candidates[0]) < np.pi / 2 else float(psi2_candidates[1])

        delta_candidates = self.delta_ff(kappa, psi2_des).ravel()
        delta_ff = float(delta_candidates[0] if abs(delta_candidates[0]) < np.pi / 2 else delta_candidates[1])

        # Build reference stack (N steps ahead)
        if traj.shape[0] < self.N + 1:
            last = np.repeat(traj[-1:, :4], self.N - (traj.shape[0] - 1), axis=0)
            ref_mat = np.vstack([traj[1:, :4], last])
        else:
            ref_mat = traj[1:self.N + 1, :4]
        X_ref = ref_mat.reshape(-1, 1)
        X0_vec = np.array([X_0, Y_0, psi1_0, psi2_0], dtype=float).reshape(nx, 1)


        try:
            # Solve QP for v
            v = self.optimization(
                A_d, B_d, W_d, U_prev_total, X0_vec, X_ref, delta_ff, print_level="NONE"
            )

        except RuntimeError as e:
            self.n_consecutive_failures += 1
            if self.n_consecutive_failures >= 20:
                print("Too many consecutive MPC failures")
                raise e
            return self.v.pop(0) if len(self.v) else U_prev_total

        self.v = []
        for i in range(len(v)):
            u = float(v[i]) + delta_ff  # optimized v + feed-forward steering
            u = float(np.clip(u, self.umin, self.umax))
            self.v.append(u)
        return self.v.pop(0)  # return first optimal input
