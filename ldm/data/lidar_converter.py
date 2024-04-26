import cv2
import random
import numpy as np


class LidarConverter:
    def __init__(
        self,
        H = 32,
        W = 1024,
        fov = (10, -30),
        depth_range = (1, 51.2),
        log_scale = True,
        depth_scale = 5.7,
    ) -> None:
        self.H = H
        self.W = W
        self.fov = fov
        self.depth_range = depth_range
        self.fov_up = fov[0] / 180.0 * np.pi
        self.fov_down = fov[1] / 180.0 * np.pi
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)
        self.base_size = (H, W)
        self.log_scale = log_scale
        self.depth_scale = depth_scale

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
            mask: np.array, shape (N,)
        """
        pcd = pcd.copy()
        label = label.copy() if label is not None else None

        # get depth (distance) of all points
        depth = np.linalg.norm(pcd, 2, axis=1)

        # mask points out of range
        mask = np.logical_and(depth > self.depth_range[0], depth < self.depth_range[1])
        depth, pcd = depth[mask], pcd[mask]

        # get scan components
        scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_range  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.W  # in [0.0, W]
        proj_y *= self.H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.maximum(0, np.minimum(self.base_size[1] - 1, np.floor(proj_x))).astype(
            np.int32
        )  # in [0,W-1]
        proj_y = np.maximum(0, np.minimum(self.base_size[0] - 1, np.floor(proj_y))).astype(
            np.int32
        )  # in [0,H-1]

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        proj_x, proj_y = proj_x[order], proj_y[order]

        # project depth
        depth = depth[order]
        range_depth = np.full(self.base_size, -1, dtype=np.float32)
        range_depth[proj_y, proj_x] = depth

        # project point range_int
        if label is not None:
            label = label[mask][order]
            range_int = np.full(self.base_size, 0, dtype=np.float32)
            range_int[proj_y, proj_x] = label
        else:
            range_int = None

        range_depth = np.where(range_depth < 0, 0, range_depth)
        if self.log_scale:
            range_depth = np.log2(range_depth + 0.0001 + 1)
            range_depth = range_depth / self.depth_scale
            range_depth = range_depth * 2.0 - 1.0
            range_depth = np.clip(range_depth, -1, 1)

        return range_depth, range_int, mask

    def range2pcd(
        self,
        range_depth,
        range_int=None,
    ):
        """
        Convert range view to point cloud

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
        Returns:
            pcd: np.array, shape (N, 3)
            label: np.array, shape (N,)
        """
        range_depth, range_int, _ = self._copy_tensors(range_depth, range_int, None)

        # derasterize with default dimensions
        range_depth, range_int, _ = self.resize(range_depth, range_int, new_H=self.base_size[0], new_W=self.base_size[1])

        if self.log_scale:
            range_depth = (range_depth + 1) / 2
            range_depth = range_depth * self.depth_scale
            range_depth = np.exp2(range_depth) - 1

        depth = range_depth.flatten()

        scan_x, scan_y = np.meshgrid(np.arange(self.base_size[1]), np.arange(self.base_size[0]))
        scan_x = scan_x.astype(np.float32) / self.base_size[1]
        scan_y = scan_y.astype(np.float32) / self.base_size[0]

        yaw = (np.pi * (scan_x * 2 - 1)).flatten()
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down)).flatten()

        pcd = np.zeros((len(yaw), 3)).astype(np.float32)
        print(yaw.shape, pitch.shape, depth.shape, pcd.shape)
        pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
        pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
        pcd[:, 2] = np.sin(pitch) * depth

        # mask out invalid points
        mask = np.logical_and(depth > self.depth_range[0], depth < self.depth_range[1])
        pcd = pcd[mask, :]

        # range_int
        if range_int is not None:
            label = range_int.flatten()[mask]
        else:
            label = None

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
        scan_x, scan_y = bbox_3d[:, 0], bbox_3d[:, 1]
        yaw = -np.arctan2(scan_y, scan_x)
        min_yaw = np.min(yaw)

        # rotate bbox_3d around z-axis by -min_yaw
        c, s = np.cos(min_yaw), np.sin(min_yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        bbox_3d = np.dot(R, bbox_3d.T).T
        scan_x, scan_y, scan_z = bbox_3d[:, 0], bbox_3d[:, 1], bbox_3d[:, 2]

        # get angles of all points
        yaw = -(np.arctan2(scan_y, scan_x) - min_yaw)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_range  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        if self.log_scale:
            depth = np.log2(depth + 0.0001 + 1)
            depth = depth / self.depth_scale
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
        bbox_range_coords=None,
        new_W=1024,
        new_H=32,
    ):
        """
        Resize range view using nearest neighbor interpolation

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            bbox_range_coords: np.array, shape (8, 3)
            new_W: int
            new_H: int
        Returns:
            range_depth: np.array, shape (new_H, new_W)
            range_int: np.array, shape (new_H, new_W)
            bbox_range_coords: np.array, shape (N, 3)
        """
        range_depth, range_int, bbox_range_coords = self._copy_tensors(
            range_depth, range_int, bbox_range_coords
        )

        if range_depth is not None and range_depth.shape != (new_H, new_W):
            range_depth = cv2.resize(
                range_depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
        if range_int is not None and range_int.shape != (new_H, new_W):
            range_int = cv2.resize(
                range_int, (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
        if bbox_range_coords is not None:
            bbox_range_coords[:, 0] = bbox_range_coords[:, 0] * new_W / self.W
            bbox_range_coords[:, 1] = bbox_range_coords[:, 1] * new_H / self.H

        self.W, self.H = new_W, new_H

        return range_depth, range_int, bbox_range_coords

    def vertical_pad(
        self,
        range_depth=None,
        range_int=None,
        bbox_range_coords=None,
        val=-1,
        pad=0,
    ):
        """
        Pad range view vertically. Pads equally on top and bottom.

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            bbox_range_coords: np.array, shape (8, 3)
            val: int, value to fill the padding
            pad: int, padding size
        Returns:
            range_depth: np.array, shape (H + pad, W)
            range_int: np.array, shape (H + pad, W)
            bbox_range_coords: np.array, shape (8, 3)
        """
        range_depth, range_int, bbox_range_coords = self._copy_tensors(
            range_depth, range_int, bbox_range_coords
        )
        padding = ((pad // 2, pad // 2), (0, 0))

        if range_depth is not None:
            range_depth = np.pad(
                range_depth, padding, mode="constant", constant_values=val
            )
        if range_int is not None:
            range_int = np.pad(range_int, padding, mode="constant", constant_values=val)
        if bbox_range_coords is not None:
            bbox_range_coords[:, 1] += pad // 2

        self.H += pad

        return range_depth, range_int, bbox_range_coords
    
    def undo_vertical_pad(
        self,
        range_depth=None,
        range_int=None,
        bbox_range_coords=None,
        pad=0,
    ):
        """
        Unpad range view vertically. Removes padding equally from top and bottom.

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            bbox_range_coords: np.array, shape (8, 3)
            pad: int, padding size
        Returns:
            range_depth: np.array, shape (H - pad, W)
            range_int: np.array, shape (H - pad, W)
            bbox_range_coords: np.array, shape (8, 3)
        """
        range_depth, range_int, bbox_range_coords = self._copy_tensors(
            range_depth, range_int, bbox_range_coords
        )

        if range_depth is not None:
            range_depth = range_depth[pad // 2 : -pad // 2]
        if range_int is not None:
            range_int = range_int[pad // 2 : -pad // 2]
        if bbox_range_coords is not None:
            bbox_range_coords[:, 1] -= pad // 2

        self.H -= pad

        return range_depth, range_int, bbox_range_coords

    def tile(
        self,
        range_depth=None,
        range_int=None,
        bbox_range_coords=None,
        n=3,
    ):
        """
        Tile range view horizontally

        Args:
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
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
        if bbox_range_coords is not None:
            bbox_range_coords[:, 0] += self.W

        self.W *= n

        return range_depth, range_int, bbox_range_coords

    def bbox_crop(
        self,
        bbox_range_coords,
        range_depth=None,
        range_int=None,
        image_width=512,
        crop_left=None,
    ):
        """
        Crop range view using bbox coordinates

        Args:
            bbox_range_coords: np.array, shape (8, 3)
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            image_width: int
        Returns:
            range_depth: np.array, shape (H, image_width)
            range_int: np.array, shape (H, image_width)
            bbox_range_coords: np.array, shape (8, 3)
            crop_left: int, left-side crop from range view
        """
        assert bbox_range_coords is not None
        range_depth, range_int, bbox_range_coords = self._copy_tensors(
            range_depth, range_int, bbox_range_coords
        )

        center_x = int(np.mean(bbox_range_coords[:, 0]))
        if crop_left is None:
            d_left = random.randint(image_width // 4, image_width - image_width // 4)
        else:
            d_left = center_x - crop_left
        d_right = image_width - d_left

        if range_depth is not None:
            range_depth = range_depth[:, center_x - d_left : center_x + d_right]
        if range_int is not None:
            range_int = range_int[:, center_x - d_left : center_x + d_right]
        bbox_range_coords = bbox_range_coords - np.array([center_x - d_left, 0, 0])

        crop_left = center_x - d_left
        return range_depth, range_int, bbox_range_coords, crop_left

    def _copy_tensors(
        self,
        range_depth=None,
        range_int=None,
        bbox_range_coords=None,
    ):
        """
        Utility function to copy tensors
        """
        range_depth = range_depth.copy() if range_depth is not None else None
        range_int = range_int.copy() if range_int is not None else None
        bbox_range_coords = (
            bbox_range_coords.copy() if bbox_range_coords is not None else None
        )

        return range_depth, range_int, bbox_range_coords

    def apply_default_transforms(
        self,
        bbox_range_coords,
        range_depth=None,
        range_int=None,
        image_height=512,
        image_width=512,
        crop_left=None,
    ):
        """
        Apply default transforms to range view which includes resizing, padding, tiling and cropping

        Args:
            bbox_range_coords: np.array, shape (8, 3)
            range_depth: np.array, shape (H, W)
            range_int: np.array, shape (H, W)
            image_height: int
            image_width: int
            crop_left: int, left-side crop from range view
        Returns:
            range_depth: np.array, shape (image_height, image_width)
            range_int: np.array, shape (image_height, image_width)
            bbox_range_coords: np.array, shape (8, 3)
        """
        range_depth, range_int, bbox_range_coords = self.resize(
            range_depth, range_int, bbox_range_coords, new_H=image_height // 2
        )
        range_depth, range_int, bbox_range_coords = self.vertical_pad(
            range_depth, range_int, bbox_range_coords, pad=image_height // 2
        )
        range_depth, range_int, bbox_range_coords = self.tile(
            range_depth, range_int, bbox_range_coords, n=3
        )
        range_depth, range_int, bbox_range_coords, crop_left = self.bbox_crop(
            bbox_range_coords, range_depth, range_int, image_width=image_width, crop_left=crop_left
        )

        return range_depth, range_int, bbox_range_coords, crop_left
    
    def undo_default_transforms(
        self,
        crop_left,
        range_depth_crop=None,
        range_int_crop=None,
        range_depth=None,
        range_int=None,
        image_height=512,
        image_width=512,
    ):
        assert range_depth is None or range_depth_crop is not None, "If range_depth is not None, range_depth_crop must be provided"
        assert range_int is None or range_int_crop is not None, "If range_int is not None, range_int_crop must be provided"

        range_depth, range_int, _ = self._copy_tensors(range_depth, range_int)
        range_depth_crop, range_int_crop, _ = self._copy_tensors(range_depth_crop, range_int_crop)

        range_depth_crop, range_int_crop, _ = self.undo_vertical_pad(
            range_depth_crop, range_int_crop, pad=image_height // 2
        )

        range_depth_crop, range_int_crop, _ = self.resize(
            range_depth_crop, range_int_crop, new_W=image_width, new_H=range_depth.shape[0]
        )

        if range_depth is not None:
            right = min(crop_left + range_depth_crop.shape[1], range_depth.shape[1])
            range_depth[:, crop_left : crop_left + right] = range_depth_crop[:, : right - crop_left]
            range_depth[:, :image_width - (right - crop_left)] = range_depth_crop[:, right - crop_left :]
        if range_int is not None:
            right = min(crop_left + range_int_crop.shape[1], range_int.shape[1])
            range_int[:, crop_left : crop_left + right] = range_int_crop[:, : right - crop_left]
            range_int[:, :image_width - (right - crop_left)] = range_int_crop[:, right - crop_left :]

        return range_depth, range_int
