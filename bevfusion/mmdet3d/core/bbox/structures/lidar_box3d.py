import numpy as np
import torch

from mmdet3d.core.points import BasePoints
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from .base_box3d import BaseInstance3DBoxes
from .utils import limit_period, rotation_3d_in_axis


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                            up z    x front (yaw=-0.5*pi)
                               ^   ^
                               |  /
                               | /
      (yaw=-pi) left y <------ 0 -------- (yaw=0)

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and decreases from
    the negative direction of y to the positive direction of x.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y<-------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)
        """
        # TODO: rotation_3d_in_axis function do not support
        #  empty tensor currently.
        assert len(self.tensor) != 0
        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        ).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # Change relative origin to [0.5, 0.5, 0.5] as dataloader uses nuscenes format
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=2)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation
        in XYWHR format."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self):
        """torch.Tensor: A tensor of 2D BEV box of each box
        without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(
            conditions, bev_rotated_boxes[:, [0, 1, 3, 2]], bev_rotated_boxes[:, :4]
        )

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns \
                None, otherwise it returns the rotated points and the \
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        assert (
            angle.shape == torch.Size([3, 3]) or angle.numel() == 1
        ), f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            rot_sin = torch.sin(angle)
            rot_cos = torch.cos(angle)
            rot_mat_T = self.tensor.new_tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[1, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)

        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                # clockwise
                points.rotate(-angle)
            else:
                raise ValueError
            return points, rot_mat_T
        else:
            return rot_mat_T

    def flip(self, bev_direction="horizontal", points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, None):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == "vertical":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 1] = -points[:, 1]
                elif bev_direction == "vertical":
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | torch.Tensor): the range of box
                (x_min, y_min, x_max, y_max)

        Note:
            The original implementation of SECOND checks whether boxes in
            a range by checking whether the points are in a convex
            polygon, we reduce the burden for simpler cases.

        Returns:
            torch.Tensor: Whether each box is inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > box_range[0])
            & (self.tensor[:, 1] > box_range[1])
            & (self.tensor[:, 0] < box_range[2])
            & (self.tensor[:, 1] < box_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): the target Box mode
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: \
                The converted box of the same type in the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode

        return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def points_in_boxes(self, points):
        """Find the box which the points are in.

        Args:
            points (torch.Tensor): Points in shape (N, 3).

        Returns:
            torch.Tensor: The index of box where each point are in.
        """
        box_idx = points_in_boxes_gpu(
            points.unsqueeze(0), self.tensor.unsqueeze(0).to(points.device)
        ).squeeze(0)
        return box_idx
