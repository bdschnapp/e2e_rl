import scipy.signal as signal
from e2erl_utils.config import *


class BasicVehicleModel:
    def __init__(self, args=None):
        # Define numeric values for substitution
        self.m = 1500  # Mass of the vehicle (kg)
        self.Iz = 3000  # Moment of inertia (kg.m^2)
        self.Cf = 80000  # Cornering stiffness of front tires (N/rad)
        self.Cr = 80000  # Cornering stiffness of rear tires (N/rad)
        self.lf = 1.2  # Distance from CG to front axle (m)
        self.lr = 1.6  # Distance from CG to rear axle (m)
        self.rho = 1.225  # Density of air (kg / m^3)
        self.Cd = 0.208  # Drag Coefficient
        self.A = 2.4  # Drag Cross-Section area (m^2)
        self.xd = 1  # Initial Longitudinal velocity (m/s)
        self.dt = 0.1  # Sampling time (s)

        if args:
            self.m = args['m']
            self.Iz = args['Iz']
            self.Cf = args['Cf']
            self.Cr = args['Cr']
            self.lf = args['lf']
            self.lr = args['lr']
            self.Cd = args['Cd']
            self.A = args['A']
            self.dt = args['dt']

        # state variables
        self.xdd = 0
        self.x = 0
        self.ydd = 0
        self.yd = 0
        self.y = 0
        self.p = 0
        self.pd = 0
        self.pdd = 0

        # control signals
        self.s = 0
        self.Fx = 0

        # longitudinal PID controller variables
        self.kp = 3
        self.ki = 0.3
        self.kd = 0.05
        self.i = 0
        self.e_prev = 0

    def reset(self, xd, x=0, y=0, p=0):
        self.xdd = 0
        self.xd = xd
        self.x = x
        self.ydd = 0
        self.yd = 0
        self.y = y
        self.pdd = 0
        self.pd = 0
        self.p = p
        self.s = 0
        self.Fx = 0

    def longitudinal_PID_controller(self, r):
        e = r - self.xd
        p = self.kp * e
        self.i += e * self.dt
        i = self.ki * self.i
        d = self.kd * (e - self.e_prev) / self.dt
        xdd = p + i + d
        self.Fx = self.m * xdd
        return self.Fx

    def loop(self, action):
        raise NotImplementedError


class DynamicsVehicleModel(BasicVehicleModel):
    def loop(self, action, override_steering_angle=None):
        self.s = np.clip(self.s + action[0] * self.dt, -np.pi / 4, np.pi / 4)
        if override_steering_angle:
            self.s = np.clip(override_steering_angle[0], -np.pi / 4, np.pi / 4)

        self.longitudinal_PID_controller(action[1])

        Fyf = self.Cf * (self.s - ((self.yd + (self.lf * self.pd))/self.xd))
        Fyr = self.Cr * (-1 * ((self.yd - (self.lr * self.pd))/self.xd))
        # can add front/rear wheel control here later
        Fxf = self.Fx / 2
        Fxr = self.Fx / 2

        self.xdd = ((Fxr + (Fxf * np.cos(self.s)) - (Fyf * np.sin(self.s))) / self.m) + (self.pd * self.yd)
        self.ydd = ((Fyr + (Fxf * np.sin(self.s)) - (Fyf * np.cos(self.s))) / self.m) - (self.pd * self.yd)
        self.pdd = ((self.lf * Fxf * np.sin(self.s)) + (self.lr * Fyf * np.cos(self.s)) - (self.lr * Fyr)) / self.Iz

        self.pd += self.pdd * self.dt
        self.p += self.pd * self.dt

        self.xd += self.xdd * self.dt
        if self.xd <= 0:
            self.xd = 0.1
        self.x += self.xd * self.dt * np.cos(self.p) - self.yd * self.dt * np.sin(self.p)

        self.yd += self.ydd * self.dt
        self.y += self.yd * self.dt * np.cos(self.p) + self.xd * self.dt * np.sin(self.p)

        return np.array([
            self.x,
            self.y,
            self.xd,
            self.yd,
            self.p,
            self.pd,
            self.s
        ], dtype='object')


class StateSpaceVehicleModel(BasicVehicleModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.dt = 0.1

    def _get_ct_matrices(self):
        s = self.s if self.xd > 0 else -self.s
        safe_xd = abs(self.xd) if abs(self.xd) > 0.01 else 0.01

        """ Return the continuous-time system matrices A and B """
        A11 = -1 * (self.Cd * self.rho * self.A * safe_xd) / (2 * self.m)
        A12 = (self.Cf * np.sin(s)) / (self.m * safe_xd)
        A13 = (self.Cf * np.sin(s) * self.lf) / (self.m * safe_xd)
        A22 = -1 * (self.Cr + (self.Cf * np.cos(s))) / (self.m * safe_xd)
        A23 = ((self.Cr * self.lr) - (self.Cf * self.lf * np.cos(s))) / (self.m * safe_xd)
        A32 = ((self.Cr * self.lr) - (self.Cf * self.lf * np.cos(s))) / (self.Iz * safe_xd)
        A33 = -1 * ((self.Cr * np.square(self.lr)) + (self.Cf * np.square(self.lf) * np.cos(s))) / (self.Iz * safe_xd)
        A = np.array([
            [A11, A12, A13],
            [0,   A22, A23],
            [0,   A32, A33]
        ])

        B11 = (1 + np.cos(s)) / (2 * self.m)
        # B12 = -1 * (self.Cf * np.sin(s) / self.m)
        B12 = 0
        B21 = np.sin(s) / (2 * self.m)
        B22 = -1 * (self.Cf * np.cos(s) / self.m)
        B31 = (self.lf * np.sin(s)) / (2 * self.Iz)
        B32 = (self.lf * self.Cf * np.cos(s)) / self.Iz
        B = np.array([
            [B11, B12],
            [B21, B22],
            [B31, B32]
        ])

        return A, B

    def _expensive_discretization(self, A, B, method):
        sys_continuous = (A, B, np.eye(A.shape[0]), 0)
        sys_discrete = signal.cont2discrete(sys_continuous, self.dt, method=method)
        A_d, B_d = sys_discrete[0], sys_discrete[1]
        return A_d, B_d

    @staticmethod
    def check_stability(A_c, A_d):
        eig_c = np.linalg.eigvals(A_c)
        eig_d = np.linalg.eigvals(A_d)

        if all(np.real(eig_c) < 0) and all(np.abs(eig_d) < 1):
            return True
        return False

    def loop(self, action):
        self.s = np.clip(self.s + action[0] * self.dt, -np.pi / 4, np.pi / 4)

        # Calculate Fx
        self.longitudinal_PID_controller(action[1])

        # Get continuous-time matrices
        A_c, B_c = self._get_ct_matrices()

        # Discretize matrices with Taylor series approximation
        A_d, B_d = self._expensive_discretization(A_c, B_c, 'zoh')

        state = np.array([self.xd, self.yd, self.pd])
        u = np.array([self.Fx, self.s])
        state_dot = A_d @ state + B_d @ u

        self.xd = state_dot[0]
        self.yd = state_dot[1]
        self.pd = state_dot[2]

        self.p += self.pd * self.dt
        self.x += self.xd * self.dt * np.cos(self.p) - self.yd * self.dt * np.sin(self.p)
        self.y += self.yd * self.dt * np.cos(self.p) + self.xd * self.dt * np.sin(self.p)

        return np.array([
            self.x,
            self.y,
            self.xd,
            self.yd,
            self.p,
            self.pd,
            self.s
        ], dtype='object')
