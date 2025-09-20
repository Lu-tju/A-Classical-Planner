#!/usr/bin/env python3
import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

def load_ply_to_pointcloud2(ply_file, frame_id="map", z_limit=3.0):
    # 使用 open3d 读取 ply 文件
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # 只保留 z < z_limit 的点
    mask = points[:, 2] < z_limit
    points = points[mask]

    rospy.loginfo(f"Loaded {points.shape[0]} points under z < {z_limit}m")

    # 生成 PointCloud2
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]

    return pc2.create_cloud(header, fields, points)

def main():
    rospy.init_node("ply_pointcloud_publisher", anonymous=True)
    pub = rospy.Publisher("/ply_pointcloud", PointCloud2, queue_size=1)

    ply_path = rospy.get_param("~ply_path", "map.ply")  # 默认路径，可通过 launch 传参
    frame_id = rospy.get_param("~frame_id", "world")
    z_limit = rospy.get_param("~z_limit", 3.5)  # 默认 3m

    rospy.loginfo(f"Loading point cloud from: {ply_path}")
    cloud_msg = load_ply_to_pointcloud2(ply_path, frame_id, z_limit)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        cloud_msg.header.stamp = rospy.Time.now()
        pub.publish(cloud_msg)
        rate.sleep()

if __name__ == "__main__":
    main()
