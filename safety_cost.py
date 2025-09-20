import os
import numpy as np
import open3d as o3d
import casadi as ca
from scipy.ndimage import distance_transform_edt


class ESDFCost:
    def __init__(self, file):
        self.map_expand_min = np.array([0.0, 0.0, 0.0])
        self.map_expand_max = np.array([0.0, 0.0, 0.0])
        self.voxel_size = 0.2
        self.min_bound = None
        self.max_bound = None
        self.sdf_shape = None
        self.sdf_map = self.get_sdf_from_ply(file)
        self.build_interpolant()

    def query_dist(self, xyz):
        """
        xyz: MX 或 DM 3x1
        """
        i, j, k = self.get_voxel_index(xyz)
        dist = self.get_esdf(ca.vertcat(i, j, k))
        return dist

    def get_voxel_index(self, xyz):
        x, y, z = xyz[0], xyz[1], xyz[2]
        i = (x - self.min_bound[0]) / self.voxel_size
        j = (y - self.min_bound[1]) / self.voxel_size
        k = (z - self.min_bound[2]) / self.voxel_size
        # clamp
        i = ca.fmin(ca.fmax(i, 0), self.sdf_shape[0] - 2)
        j = ca.fmin(ca.fmax(j, 0), self.sdf_shape[1] - 2)
        k = ca.fmin(ca.fmax(k, 0), self.sdf_shape[2] - 2)
        return i, j, k

    def build_interpolant(self, method="linear"):
        Nx, Ny, Nz = self.sdf_shape
        grid_x = np.arange(Nx)
        grid_y = np.arange(Ny)
        grid_z = np.arange(Nz)

        # flatten (Fortran order)
        sdf_flat = self.sdf_map.ravel(order="F")

        # CasADi三维插值
        self.get_esdf = ca.interpolant(
            "get_esdf",
            method,
            [grid_x, grid_y, grid_z],
            sdf_flat
        )

    def get_sdf_from_ply(self, file):
        pcd = o3d.io.read_point_cloud(file)
        self.min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
        self.max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
        points = np.asarray(pcd.points)
        print(
            f"{os.path.basename(file)}: x=({self.min_bound[0] + self.map_expand_min[0]:.2f}, {self.max_bound[0] - self.map_expand_max[0]:.2f}), "
            f"y=({self.min_bound[1] + self.map_expand_min[1]:.2f}, {self.max_bound[1] - self.map_expand_max[1]:.2f}), "
            f"z=({self.min_bound[2] + self.map_expand_min[2]:.2f}, {self.max_bound[2] - self.map_expand_max[2]:.2f})")

        self.sdf_shape = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)
        voxel_indices = ((points - self.min_bound) / self.voxel_size).astype(int)

        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < self.sdf_shape), axis=1)
        voxel_indices = voxel_indices[valid_mask]

        occupancy = np.zeros(self.sdf_shape, dtype=np.uint8)
        occupancy[tuple(voxel_indices.T)] = 1

        obstacle_mask = occupancy == 1
        free_mask = occupancy == 0

        dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
        dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

        dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

        return dist_to_obstacle  # shape: (X, Y, Z)


if __name__ == '__main__':
    cost = ESDFCost("map.ply")
