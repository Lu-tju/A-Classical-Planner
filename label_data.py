import numpy as np
from scipy.spatial.transform import Rotation as R
from trajectory_optimizer import TrajectoryOptimizer
from config import cfg
from tqdm import tqdm

class DataGenerator:
    def __init__(self, optimizer):
        """
        :param optimizer: 你的优化器对象，包含 init_state() 和 optimize()
        """
        self.optimizer = optimizer
        self.Tf = cfg.Tf
        self.goal_pitch_std = 10.0
        self.goal_yaw_std = 20.0
        self.goal_length = cfg.r
        self.a_mean = np.array([0, 0, 0])
        self.a_std = np.array([0.5, 0.5, 0.3])
        self.v_mean = np.array([0.4, 0, 0])
        self.v_std = np.array([2, 0.45, 0.3])
        self.vx_lognorm_mean = np.log(1 - 0.4)
        self.vx_logmorm_sigma = np.log(2.0)
        self.vel_max = cfg.v_max
        self.acc_max = cfg.a_max

    def load_dataset(self, data_path):
        states = np.loadtxt(data_path, delimiter=',', skiprows=1).astype(np.float32)
        self.p0 = states[:, 0:3]
        self.q0 = states[:, 3:7]
        self.dataset_size = self.p0.shape[0]

    def generate_states(self, p0, q0):
        vel_b, acc_b = self._get_random_state()

        # generate random goal in front of the quadrotor.
        R_WB = R.from_quat([q0[1], q0[2], q0[3], q0[0]])  # q: wxyz
        euler_angles = R_WB.as_euler('ZYX', degrees=False)  # [yaw(z) pitch(y) roll(x)]
        R_Ww = R.from_euler('ZYX', [euler_angles[0], 0, 0], degrees=False)
        goal_w = self._get_random_goal()
        goal_W = R_Ww.apply(goal_w) + p0
        goal_b = R_WB.inv().apply(goal_W - p0)

        vel_w = R_WB.apply(vel_b)
        acc_w = R_WB.apply(acc_b)
        return vel_w, acc_w, goal_W, vel_b, acc_b, goal_b, R_WB

    def _get_random_state(self):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:  # avoid outliers
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self):
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        random_near = np.random.rand()
        if random_near < 0.1:
            goal_w_dir = random_near * 10 * goal_w_dir
        return self.goal_length * goal_w_dir

    def generate_and_save(self, npz_path):
        """
        :param p0, v0, a0, goal: numpy arrays of shape (N, 3)
        :param npz_path: 保存的 npz 文件路径
        """
        # 预分配结果
        v0_all = np.zeros((self.dataset_size, 3))
        a0_all = np.zeros((self.dataset_size, 3))
        p1_all = np.zeros((self.dataset_size, 3))
        v1_all = np.zeros((self.dataset_size, 3))
        a1_all = np.zeros((self.dataset_size, 3))
        Tf_all = np.full((self.dataset_size, 1), self.Tf)
        goal_all = np.zeros((self.dataset_size, 3))

        for i in tqdm(range(self.dataset_size)):
            # 初始化优化器
            vel_w0, acc_w0, goal_w, vel_b0, acc_b0, goal_b, R_wb = self.generate_states(self.p0[i], self.q0[i])
            self.optimizer.init_state(self.p0[i], vel_w0, acc_w0, goal_w)

            # 调用优化器
            p1, v1, a1 = self.optimizer.optimize(x1=None)

            p1_b = R_wb.inv().apply(p1 - self.p0[i])
            v1_b = R_wb.inv().apply(v1)
            a1_b = R_wb.inv().apply(a1)
            v0_all[i] = vel_b0
            a0_all[i] = acc_b0
            p1_all[i] = p1_b
            v1_all[i] = v1_b
            a1_all[i] = a1_b
            goal_all[i] = goal_b

        # 保存到 npz
        np.savez(
            npz_path,
            p0=self.p0,
            q0=self.q0,
            v0=v0_all,
            a0=a0_all,
            goal=goal_all,
            p1=p1_all,
            v1=v1_all,
            a1=a1_all,
            Tf=Tf_all
        )

        print(f"Saved {self.dataset_size} samples to {npz_path}")


def plot_hist(npz_path):
    import matplotlib.pyplot as plt

    # 加载 npz 文件
    data = np.load(npz_path)

    v0 = data["v0"]
    a0 = data["a0"]
    goal = data["goal"]
    p1 = data["p1"]
    v1 = data["v1"]
    a1 = data["a1"]

    # 要绘制直方图的变量
    variables = {
        "v0": v0,
        "a0": a0,
        "goal": goal,
        "p1": p1,
        "v1": v1,
        "a1": a1
    }

    for name, arr in variables.items():
        plt.figure(figsize=(12, 4))
        for dim in range(3):
            plt.subplot(1, 3, dim + 1)
            plt.hist(arr[:, dim], bins=50, alpha=0.7)
            plt.title(f"{name} - {'xyz'[dim]}")
            plt.grid(True)
        plt.suptitle(f"{name} distribution")
        plt.tight_layout()
        plt.show()


# ===== 使用示例 =====
if __name__ == "__main__":
    import os, time
    start = time.time()
    base_path = "/home/lu/YOPO/dataset/"
    num_dirs = len([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for dataset_id in range(num_dirs):
        print(f"Labeling the {dataset_id} dataset...")
        optimizer = TrajectoryOptimizer(base_path + f"pointcloud-{dataset_id}.ply", verbose=False)
        gen = DataGenerator(optimizer)
        gen.load_dataset(base_path + f"pose-{dataset_id}.csv")
        gen.generate_and_save(base_path + f"label-{dataset_id}.npz")

    print("Label Time:", time.time() - start)
    # plot_hist(base_path + f"label-{dataset_id}.npz")

