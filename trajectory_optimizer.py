import casadi as ca
import numpy as np
from poly_solver import Poly5Solver
from safety_cost import ESDFCost
from config import cfg

class TrajectoryOptimizer:
    def __init__(self, map_path, verbose=True):
        self.safety_cost = ESDFCost(map_path)
        self.Tf = cfg.Tf
        self.N = 10
        self.d0 = 1.2
        self.r = 0.4
        self.verbose = verbose

    def init_state(self, p0, v0, a0, goal):
        self.p0, self.v0, self.a0 = p0, v0, a0
        self.goal = goal

    def optimize(self, x1=None):
        """
        Args:
            x1: 世界系的优化初值 [p1(3), v1(3), a1(3)]
        """

        # --- 符号变量 ---
        p1 = ca.MX.sym("p1", 3)
        v1 = ca.MX.sym("v1", 3)
        a1 = ca.MX.sym("a1", 3)

        # --- 构建多项式 ---
        poly = Poly5Solver(self.p0, self.v0, self.a0, p1, v1, a1, self.Tf)

        # --- 离散积分代价 ---
        dt = self.Tf / self.N
        cost = 0
        for i in range(self.N + 1):
            t_i = i * dt
            pos_i = poly.get_position(t_i)
            jerk_i = poly.get_jerk(t_i)

            # 安全代价
            dist = self.safety_cost.query_dist(pos_i)
            safe_cost = cfg.safety_weight * ca.exp(-(dist - self.d0) / self.r)

            # 平滑代价
            smooth_cost = cfg.smoothness_weight * ca.sumsqr(jerk_i)

            cost += dt * (safe_cost + smooth_cost)

        # 末态到目标的距离
        cost += cfg.goal_weight * ca.sumsqr(p1 - self.goal)

        # --- NLP 设置 ---
        x = ca.vertcat(p1, v1, a1)

        # --- 约束 g(x) ---
        # 末态与初态距离约束: ||p1 - p0||^2 <= r^2
        g = ca.sumsqr(p1 - self.p0)

        nlp = {"x": x, "f": cost, "g": g}
        solver = ca.nlpsol("solver", "ipopt", nlp, {
            "ipopt.print_level": 0,
            "print_time": self.verbose,  # 静默1: 禁止 CasADi 打印 solver 时间
            "ipopt.sb": "yes",  # 静默2: (silent mode) 静默 ipopt banner
            "ipopt.max_cpu_time": 0.1,  # 限制求解时间为 0.5 秒
            "ipopt.tol": 1e-3,  # 放宽精度
            "ipopt.acceptable_tol": 1e-2,  # 可接受解
            "ipopt.max_iter": 50  # 限制迭代次数
        })

        # --- 变量边界 ---
        lbx = np.hstack([-np.inf * np.ones(3), -cfg.v_max * np.ones(3), -cfg.a_max * np.ones(3)])
        ubx = np.hstack([np.inf * np.ones(3), cfg.v_max * np.ones(3), cfg.a_max * np.ones(3)])

        # --- 约束边界 ---
        lbg = [0]  # g >= 0
        ubg = [cfg.r ** 2]  # g <= r^2

        # --- 初值 ---
        if x1 is None:
            x1 = np.hstack([self.goal, np.zeros(3), np.zeros(3)])  # p1 = self.goal, v1 = 0, a1 = 0

        sol = solver(x0=x1, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        sol_pva = sol["x"].full().flatten()
        return sol_pva[:3], sol_pva[3:6], sol_pva[6:9]
