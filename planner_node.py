#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from trajectory_optimizer import TrajectoryOptimizer
from poly_solver import *
from config import cfg
from PositionCommand import PositionCommand

class PlannerNode:
    def __init__(self, map_path, goal):
        rospy.init_node("trajectory_optimizer_node")

        # --- Trajectory Optimizer ---
        self.optimizer = TrajectoryOptimizer(map_path)
        self.goal = np.array(goal)

        # --- Current state ---
        self.p0 = np.zeros(3)
        self.v0 = np.zeros(3)
        self.a0 = np.zeros(3)
        self.odom_received = False

        # --- Subscriber ---
        rospy.Subscriber("/sim/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1)

        # --- Publisher ---
        self.cmd_pub = rospy.Publisher("/so3_control/pos_cmd", PositionCommand, queue_size=1)
        self.path_pub = rospy.Publisher("/path", Path, queue_size=1)
        # --- Optimized polynomial ---
        self.poly = None
        self.Tf = cfg.Tf
        self.t_start = None

        # --- Timers ---
        rospy.Timer(rospy.Duration(0.1), self.optimize_callback)   # 10Hz
        rospy.Timer(rospy.Duration(0.02), self.publish_callback)   # 50Hz

        rospy.loginfo("Trajectory Node initialized.")
        rospy.spin()

    def goal_callback(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    def odom_callback(self, msg):
        self.p0 = np.array([msg.pose.pose.position.x,
                            msg.pose.pose.position.y,
                            msg.pose.pose.position.z])
        self.v0 = np.array([msg.twist.twist.linear.x,
                            msg.twist.twist.linear.y,
                            msg.twist.twist.linear.z])
        self.odom_received = True

    def optimize_callback(self, event):
        if not self.odom_received:
            return

        # 初始化状态
        goal_len = np.linalg.norm(self.goal - self.p0)
        if goal_len > cfg.r:
            goal = cfg.r * (self.goal - self.p0) / goal_len + self.p0
        else:
            goal = self.goal
        self.optimizer.init_state(self.p0, self.v0, self.a0, goal)

        # 调用优化器
        p1, v1, a1 = self.optimizer.optimize(x1=None)

        # 构建多项式轨迹
        self.poly = Poly5SolverNumpy(self.p0, self.v0, self.a0, p1, v1, a1, self.Tf)
        self.t_start = rospy.Time.now()

        self.visualize_path()
        rospy.loginfo("Trajectory optimized. End position: {}".format(p1))

    def publish_callback(self, event):
        if self.poly is None or self.t_start is None:
            return

        # 当前时间
        t_now = (rospy.Time.now() - self.t_start).to_sec()
        if t_now > self.Tf:
            t_now = self.Tf  # 不超过终点

        # 获取多项式状态
        pos = self.poly.get_position(t_now)
        vel = self.poly.get_velocity(t_now)
        acc = self.poly.get_acceleration(t_now)

        self.a0 = acc
        # 发布到控制话题
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
        control_msg.position.x = pos[0]
        control_msg.position.y = pos[1]
        control_msg.position.z = pos[2]
        control_msg.velocity.x = vel[0]
        control_msg.velocity.y = vel[1]
        control_msg.velocity.z = vel[2]
        control_msg.acceleration.x = acc[0]
        control_msg.acceleration.y = acc[1]
        control_msg.acceleration.z = acc[2]
        control_msg.yaw = 0.0
        control_msg.yaw_dot = 0.0
        self.cmd_pub.publish(control_msg)

    def visualize_path(self):
        """
        poly: Poly5Solver 对象
        path_pub: rospy.Publisher 对象 (nav_msgs/Path)
        Tf: 轨迹总时长
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "world"  # 根据你的坐标系调整

        # 离散采样 10 个点
        ts = np.linspace(0, self.Tf, 10)
        for t in ts:
            pos = self.poly.get_position(t)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "world"
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.w = 1.0  # 默认方向，无旋转
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

if __name__ == "__main__":
    map_path = "map.ply"
    goal = [50.0, 0.0, 2.0]
    node = PlannerNode(map_path, goal)
