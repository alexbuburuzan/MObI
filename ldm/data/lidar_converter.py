import cv2
import random
import numpy as np


class LidarConverter:
    def __init__(
        self,
        H = 32,
        W = 1096,
        depth_interval = (1.4, 54),
        log_scale = False,
        depth_scale = 5.8,
    ) -> None:
        self.H = H
        self.W = W
        self.depth_interval = depth_interval
        self.base_size = (H, W)
        self.log_scale = log_scale
        self.depth_scale = depth_scale
        self.beam_pitch_angles = np.array([0.0232 * x for x in range(-23, 9)])

    def pcd2range(
        self,
        pcd,
        label=None,
    ):
        """
        Convert point cloud to range view

        Args:
            pcd: np.array, shape (N, 3)
            label: np.array, shape (N,)
        Returns:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            filtered_points: np.array, shape (N,)
            range_pitch: np.array, shape (H, W)
            range_yaw: np.array, shape (H, W)
        """
        pcd, label = self._copy_arrays(pcd, label)

        # get depth (distance) of all points
        depth = np.linalg.norm(pcd, 2, axis=1)

        # mask points out of range
        filtered_points = np.logical_and(depth > self.depth_interval[0], depth < self.depth_interval[1])
        depth, pcd = depth[filtered_points], pcd[filtered_points]

        # get scan components
        scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        proj_y = (pitch - self.beam_pitch_angles.min()) / (self.beam_pitch_angles.max() - self.beam_pitch_angles.min()) * 31
        proj_y = 31 - np.round(np.clip(proj_y, 0, 31)).astype(np.int32)

        # get projections in range coords
        proj_x = 0.5 * (yaw / np.pi + 1.0) * self.W
        proj_x = np.maximum(0, np.minimum(self.base_size[1] - 1, np.floor(proj_x))).astype(np.int32)

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        proj_x, proj_y = proj_x[order], proj_y[order]
        depth, pitch, yaw = depth[order], pitch[order], yaw[order]

        # default yaw
        scan_x = np.meshgrid(np.arange(self.base_size[1]), np.arange(self.base_size[0]))[0]
        scan_x = scan_x.astype(np.float32) / self.base_size[1]
        range_yaw = (np.pi * (scan_x * 2 - 1))

        # default pitch
        range_pitch = np.zeros(self.base_size, dtype=np.float32)
        for i in range(32):
            range_pitch[i, :] = self.beam_pitch_angles[31 - i]

        # default depth
        range_depth = np.full(self.base_size, -1, dtype=np.float32)

        # project points to range view
        range_depth[proj_y, proj_x] = depth
        range_pitch[proj_y, proj_x] = pitch
        range_yaw[proj_y, proj_x] = yaw

        if label is not None:
            label = label[filtered_points][order]
            range_int = np.full(self.base_size, 0, dtype=np.float32)
            range_int[proj_y, proj_x] = label
        else:
            range_int = None

        range_depth = np.where(range_depth < 0, 0, range_depth)
        if self.log_scale:
            range_depth = np.log2(range_depth + 0.0001 + 1)
            range_depth = range_depth / self.depth_scale
        else:
            range_depth = range_depth / self.depth_interval[1]

        range_depth = range_depth * 2.0 - 1.0
        range_depth = np.clip(range_depth, -1, 1)

        return range_depth, range_int, filtered_points, range_pitch, range_yaw

    def range2pcd(
        self,
        range_depth,
        range_pitch,
        range_yaw,
        label=None,
    ):
        """
        Convert range view to point cloud

        Args:
            range_depth: np.array, shape (H, W)
            range_pitch: np.array, shape (H, W)
            range_yaw: np.array, shape (H, W)
            label: np.array, shape (H, W)
        Returns:
            pcd: np.array, shape (N, 3)
            label: np.array, shape (N,)
        """
        range_depth, label = self._copy_arrays(range_depth, label)

        # derasterize with default dimensions
        range_depth, label, _, _ = self.resize(
            range_depth, label, new_H=self.base_size[0], new_W=self.base_size[1]
        )
        range_depth = (range_depth + 1) / 2

        if self.log_scale:
            range_depth = range_depth * self.depth_scale
            range_depth = np.exp2(range_depth) - 1
        else:
            range_depth = range_depth * self.depth_interval[1]

        depth = range_depth.flatten()
        yaw = range_yaw.flatten()
        pitch = range_pitch.flatten()

        pcd = np.zeros((len(yaw), 3)).astype(np.float32)
        pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
        pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
        pcd[:, 2] = np.sin(pitch) * depth

        # mask out invalid points
        mask = np.logical_and(depth > self.depth_interval[0], depth < self.depth_interval[1])
        pcd = pcd[mask, :]
        label = label.flatten()[mask] if label is not None else None

        return pcd, label
    
    def get_range_coords(self, bbox_3d):
        """
        Get range coordinates of the 3D bounding box

        Args:
            bbox_3d: np.array, shape (8, 3)
        Returns:
            np.array, shape (8, 3)
            Each row is the x, y, depth coordinates of the 3D bounding box corners
                in the range view.
        """
        bbox_3d = bbox_3d.copy()

        # get depth (distance) of all points
        depth = np.linalg.norm(bbox_3d, 2, axis=1)

        # get scan components
        center_x, center_y = np.mean(bbox_3d[:, 0]), np.mean(bbox_3d[:, 1])
        center_yaw = -np.arctan2(center_y, center_x)

        # rotate bbox_3d around z-axis by -min_yaw
        c, s = np.cos(center_yaw), np.sin(center_yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        bbox_3d = np.dot(R, bbox_3d.T).T
        scan_x, scan_y, scan_z = bbox_3d[:, 0], bbox_3d[:, 1], bbox_3d[:, 2]

        # get angles of all points
        yaw = -(np.arctan2(scan_y, scan_x) - center_yaw)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        
        proj_y = (pitch - self.beam_pitch_angles.min()) / (self.beam_pitch_angles.max() - self.beam_pitch_angles.min()) * 31
        proj_y = 31 - np.round(np.clip(proj_y, 0, 31)).astype(np.int32)

        # scale to image size using angular resolution
        proj_x *= self.W

        if self.log_scale:
            depth = np.log2(depth + 0.0001 + 1)
            depth = depth / self.depth_scale
        else:
            depth = depth / self.depth_interval[1]

        depth = depth * 2.0 - 1.0
        depth = np.clip(depth, -1, 1)

        coords = np.concatenate(
            [proj_x[:, None], proj_y[:, None], depth[:, None]], axis=-1
        )
        return coords

    def resize(
        self,
        range_depth=None,
        range_int=None,
        mask=None,
        bbox_range_coords=None,
        new_W=1096,
        new_H=32,
    ):
        """
        Resize range view using nearest neighbor interpolation

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            mask: np.array, shape (H, W) object isntance mask
            bbox_range_coords: np.array, shape (8, 3)
            new_W: int
            new_H: int
        Returns:
            range_depth: np.array, shape (new_H, new_W)
            range_int: np.array, shape (new_H, new_W)
            bbox_range_coords: np.array, shape (N, 3)
        """
        range_depth, range_int, mask, bbox_range_coords = self._copy_arrays(
            range_depth, range_int, mask, bbox_range_coords
        )

        if range_depth is not None and range_depth.shape != (new_H, new_W):
            range_depth = cv2.resize(
                range_depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
        if range_int is not None and range_int.shape != (new_H, new_W):
            range_int = cv2.resize(
                range_int, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
        if mask is not None and mask.shape != (new_H, new_W):
            mask = cv2.resize(
                mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
        if bbox_range_coords is not None:
            bbox_range_coords[:, 0] = bbox_range_coords[:, 0] * new_W / self.W
            bbox_range_coords[:, 1] = bbox_range_coords[:, 1] * new_H / self.H

        self.W, self.H = new_W, new_H

        return range_depth, range_int, mask, bbox_range_coords

    def tile(
        self,
        range_depth=None,
        range_int=None,
        mask=None,
        bbox_range_coords=None,
        n=3,
    ):
        """
        Tile range view horizontally

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            mask: np.array, shape (H, W) object instance mask
            bbox_range_coords: np.array, shape (8, 3)
            n: int, number of tiles
        Returns:
            range_depth: np.array, shape (H, W * n)
            range_int: np.array, shape (H, W * n)
            bbox_range_coords: np.array, shape (8, 3)
        """
        if range_depth is not None:
            range_depth = np.tile(range_depth, n)
        if range_int is not None:
            range_int = np.tile(range_int, n)
        if mask is not None:
            mask = np.tile(mask, n)
        if bbox_range_coords is not None:
            bbox_range_coords[:, 0] += self.W

        self.W *= n

        return range_depth, range_int, mask, bbox_range_coords

    def bbox_crop(
        self,
        bbox_range_coords,
        range_depth=None,
        range_int=None,
        mask=None,
        width=1096,
        random_crop=False,
        crop_left=None,
    ):
        """
        Crop range view using bbox coordinates

        Args:
            bbox_range_coords: np.array, shape (8, 3)
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            mask: np.array, shape (H, W) object instance mask
            width: int
            random_crop: bool, whether to perform random cropping
            crop_left: int, left-side crop from range view
        Returns:
            range_depth: np.array, shape (H, width)
            range_int: np.array, shape (H, width)
            bbox_range_coords: np.array, shape (8, 3)
            crop_left: int, left-side crop from range view
        """
        assert bbox_range_coords is not None
        range_depth, range_int, mask, bbox_range_coords = self._copy_arrays(
            range_depth, range_int, mask, bbox_range_coords
        )

        center_x = int(np.mean(bbox_range_coords[:, 0]))
        if crop_left is None:
            if random_crop:
                d_left = random.randint(width // 4, width - width // 4)
            else:
                d_left = width // 2
        else:
            d_left = center_x - crop_left
        d_right = width - d_left

        if range_depth is not None:
            range_depth = range_depth[:, center_x - d_left : center_x + d_right]
        if range_int is not None:
            range_int = range_int[:, center_x - d_left : center_x + d_right]
        if mask is not None:
            mask = mask[:, center_x - d_left : center_x + d_right]
        bbox_range_coords = bbox_range_coords - np.array([center_x - d_left, 0, 0])
        bbox_range_coords[:, 0]
        crop_left = center_x - d_left

        return range_depth, range_int, mask, bbox_range_coords, crop_left

    def _copy_arrays(self, *args):
        """
        Utility function to copy tensors
        """
        return (arg.copy() if arg is not None else None for arg in args)

    def apply_default_transforms(
        self,
        bbox_range_coords,
        range_depth=None,
        range_int=None,
        mask=None,
        height=32,
        width=1096,
        crop_left=None,
        random_crop=False,
    ):
        """
        Apply default transforms to range view which includes resizing, tiling and cropping

        Args:
            bbox_range_coords: np.array, shape (8, 3)
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            mask: np.array, shape (H, W) object instance mask
            height: int
            width: int
            crop_left: int, left-side crop from range view
            random_crop: bool, whether to perform random cropping
        Returns:
            range_depth: np.array, shape (height, width)
            range_int: np.array, shape (height, width)
            bbox_range_coords: np.array, shape (8, 3)
        """
        range_depth, range_int, mask, bbox_range_coords = self.resize(
            range_depth, range_int, mask, bbox_range_coords, new_H=height
        )
        range_depth, range_int, mask, bbox_range_coords = self.tile(
            range_depth, range_int, mask, bbox_range_coords, n=3
        )
        range_depth, range_int, mask, bbox_range_coords, crop_left = self.bbox_crop(
            bbox_range_coords, range_depth, range_int, mask,
            width=width, crop_left=crop_left, random_crop=random_crop
        )

        return range_depth, range_int, mask, bbox_range_coords, crop_left
    
    def undo_default_transforms(
        self,
        crop_left,
        range_depth_crop,
        range_depth,
        range_int_crop=None,
        range_int=None,
        mask=None,
    ):
        assert range_int is None or range_int_crop is not None, "If range_int is not None, range_int_crop must be provided"

        range_depth, range_int = self._copy_arrays(range_depth, range_int)
        range_depth_crop, range_int_crop = self._copy_arrays(range_depth_crop, range_int_crop)

        ignore = -1000
        width_crop = range_depth_crop.shape[-1]
        crop_left = crop_left % range_depth.shape[-1]

        if mask is not None:
            range_depth_crop[~mask] = ignore
            if range_int_crop is not None: range_int_crop[~mask] = ignore

        range_depth_crop, range_int_crop, _, _ = self.resize(
            range_depth_crop, range_int_crop, new_W=width_crop, new_H=range_depth.shape[0]
        )

        if mask is not None:
            range_depth_aux = np.zeros_like(range_depth) + ignore
        else:
            range_depth_aux = range_depth.copy()

        right = min(crop_left + range_depth_crop.shape[1], range_depth.shape[1])
        range_depth_aux[:, crop_left : right] = range_depth_crop[:, : right - crop_left]
        range_depth_aux[:, :width_crop - (right - crop_left)] = range_depth_crop[:, right - crop_left :]

        range_depth = np.where(range_depth_aux == ignore, range_depth, range_depth_aux)

        if range_int is not None:
            if mask is not None:
                range_int_aux = np.zeros_like(range_int) + ignore
            else:
                range_int_aux = range_int.copy()

            right = min(crop_left + range_int_crop.shape[1], range_int.shape[1])
            range_int_aux[:, crop_left : right] = range_int_crop[:, : right - crop_left]
            range_int_aux[:, :width_crop - (right - crop_left)] = range_int_crop[:, right - crop_left :]

            range_int = np.where(range_int_aux == ignore, range_int, range_int_aux)

        return range_depth, range_int
