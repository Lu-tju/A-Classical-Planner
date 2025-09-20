import casadi as ca
import numpy as np

class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """
        pos0, vel0, acc0, pos1, vel1, acc1: MX column vectors (3x1)
        Tf: total time
        """
        self.Tf = Tf

        # 拼接状态矩阵，每列对应一个轴 (x,y,z)
        self.State_Mat = ca.hcat([pos0, vel0, acc0, pos1, vel1, acc1]).T

        # 5阶多项式系数矩阵
        t = Tf
        self.Coef_inv = ca.DM([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1 / 2, 0, 0, 0],
            [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
            [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
            [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]
        ])
        self.Coef_inv = ca.MX(self.Coef_inv)

        # 计算多项式系数矩阵 [3, 6]
        self.A = ca.mtimes(self.Coef_inv, self.State_Mat).T

    def get_position(self, t):
        powers = ca.vcat([t**i for i in range(6)])  # [6,1]
        pos = ca.mtimes(self.A, powers)           # [3,1]
        return pos

    def get_velocity(self, t):
        powers = ca.vcat([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        vel = ca.mtimes(self.A, powers)
        return vel

    def get_acceleration(self, t):
        powers = ca.vcat([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        acc = ca.mtimes(self.A, powers)
        return acc

    def get_jerk(self, t):
        powers = ca.vcat([0, 0, 0, 6, 24*t, 60*t**2])
        jerk = ca.mtimes(self.A, powers)
        return jerk

class Poly5SolverNumpy:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ 5-th order polynomial at each Axis """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t

