from typing import Optional
import warnings

import numba
import numpy as np
from numba import errors

from mmdet3d.core.bbox import box_np_ops, LiDARInstance3DBoxes

warnings.filterwarnings("ignore", category=errors.NumbaPerformanceWarning)


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    """Rotate 2D boxes.

    Args:
        corners (np.ndarray): Corners of boxes.
        angle (float): Rotation angle.
        rot_mat_T (np.ndarray): Transposed rotation matrix.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
        clockwise (bool): Whether the corners are in clockwise order.
            Default: True.
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # Ensure numba works
            if (
                (boxes_standup[i, 0] == boxes_standup[j, 0]) &
                (boxes_standup[i, 1] == boxes_standup[j, 1]) &
                (boxes_standup[i, 2] == boxes_standup[j, 2]) &
                (boxes_standup[i, 3] == boxes_standup[j, 3])
            ):
                # Overlap of identical boxes
                ret[i, j] = ret[j, i] = True
                continue
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret

def frustum_collision_test(
    gt_frustums: np.ndarray, 
    sp_frustums: Optional[np.ndarray] = None,
    thresh: float = 0.7,
    apply_thresh: bool = True
):
    """
    Calculates frustum collision between the input frustums for original annotations,
    and the sampled boxes for copy-paste.

    Inspired from the code in PointAugmenting:
    https://github.com/VISION-SJTU/PointAugmenting/blob/main/det3d/core/sampler/sample_ops.py#L673

    (Anuj): thresh is set to 0.7 as used in the code originally by PointAugmenting.
            Have not tried to tune this.

    Frustum arrays have general shape of M x 3 x 2 x 2.
        dim-0: M number of boxes
        dim-1: r, theta, phi
        dim-2: min and max
        dim-3: (Anuj): not very clear yet? Set to -1 for default. But if the phi_max - phi_min > pi,
               then min[0] = 0, max[0] = 2 x pi. And min[1] and max[1] set to min and max of phi
               for the 8 corners. Check get_frustum() method for more details.

    Args:
        gt_frustums: frustums for original annotations (N x 3 x 2 x 2) in the datum.
        sp_frustums: frustums for the sampled boxes (K x 3 x 2 x 2) for copy-paste.
        thresh: threshold above which if the frustums collide, the sampled box is then ignored.

    Returns:
        Frustum collision matrix, shape (N + K, N + K).
        The order is first N original boxes, and then K sampled boxes, in both rows, and columns.
    """

    if sp_frustums is None:
        sp_frustums = gt_frustums
        gt_frustums_all = gt_frustums
        N = 0
        K = sp_frustums.shape[0]
    else:
        N = gt_frustums.shape[0]
        K = sp_frustums.shape[0]
        gt_frustums_all = np.concatenate([gt_frustums, sp_frustums], axis=0)

    S = np.array(
        [
            (cur_frus[1, 1, 0] - cur_frus[1, 0, 0])
            * (
                cur_frus[2, 1, 0]
                - cur_frus[2, 0, 0]
                + cur_frus[2, 1, 1]
                - cur_frus[2, 0, 1]
            )
            for cur_frus in gt_frustums_all
        ],
        dtype=np.float32,
    )
    # assert S.any() > 0
    ret = np.zeros((N + K, N + K), dtype=np.float32)
    for i in range(N + K):
        for j in range(K):
            sp_frus = (
                [sp_frustums[j, :, :, 0]]
                if sp_frustums[j, 2, 0, 1] < 0
                else [sp_frustums[j, :, :, 0], sp_frustums[j, :, :, 1]]
            )
            gt_frus = (
                [gt_frustums_all[i, :, :, 0]]
                if gt_frustums_all[i, 2, 0, 1] < 0
                else [gt_frustums_all[i, :, :, 0], gt_frustums_all[i, :, :, 1]]
            )
            iou = 0
            for cur_sp_frus in sp_frus:
                for cur_gt_frus in gt_frus:
                    coll = (
                        max(cur_sp_frus[2, 0], cur_gt_frus[2, 0])
                        < min(cur_sp_frus[2, 1], cur_gt_frus[2, 1])
                    ) and (
                        max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0])
                        < min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0])
                    )
                    if coll:
                        iou += (
                            min(cur_sp_frus[2, 1], cur_gt_frus[2, 1])
                            - max(cur_sp_frus[2, 0], cur_gt_frus[2, 0])
                        ) * (
                            min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0])
                            - max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0])
                        )
                        # assert iou > 0

            iou_per = iou / min(S[i], S[j + N])
            # assert iou_per <= 1.01
            ret[i, j + N] = iou_per
            ret[j + N, i] = iou_per

    if apply_thresh:
        ret = ret > thresh

    return ret

def get_frustum(boxes: np.ndarray) -> np.ndarray:
    """
    Computes the frustum for the input 3d boxes.

    Inspired from PointAugmenting code
    https://github.com/VISION-SJTU/PointAugmenting/blob/main/det3d/datasets/nuscenes/nusc_common.py#L486

    Args:
        boxes: 3d boxes (N boxes) for which the frustum is to be computed.

    Returns:
        The frustum (N x 3 x 2 x 2) for the input 3d boxes.
        represents N x (r, theta, phi) * (min, max) * 2
    """
    num_box = len(boxes)
    gt_box_corners = LiDARInstance3DBoxes(boxes.copy(), box_dim=9).corners.numpy().reshape(-1, 3)

    pts_rr = transform_to_spherical(gt_box_corners)
    pts_rr = pts_rr.reshape(num_box, 8, 3)

    gt_frustum = (
        np.ones([num_box, 3, 2, 2], dtype=np.float32) * -1
    )  # N * (r, theta, phi) * (min, max) * 2
    gt_frustum[:, :, :, 0] = np.stack([pts_rr.min(axis=1), pts_rr.max(axis=1)], axis=2)
    # check if the difference between max and min in phi > pi radians
    val = (gt_frustum[:, 2, 1, 0] - gt_frustum[:, 2, 0, 0]) > np.pi
    if val.any():
        idxs = np.where(val > 0)[0]
        # set the expected mins and maxs to 0 and 2 * pi
        gt_frustum[val, 2, 0, 0] = 0.0
        gt_frustum[val, 2, 1, 1] = np.pi * 2
        for idx in idxs:
            # update the actual min and max for all the 8 corners.
            # for the min, check corners with phi > pi and pick the one with minimum value
            # for the max, check corners with phi < pi and pick the one with maximum value
            # (Anuj): I don't get this as to why they do this?
            gt_frustum[idx, 2, 1, 0] = pts_rr[idx, pts_rr[idx, :, 2] < np.pi, 2].max()
            gt_frustum[idx, 2, 0, 1] = pts_rr[idx, pts_rr[idx, :, 2] > np.pi, 2].min()
    return gt_frustum


def transform_to_spherical(points: np.ndarray):
    """
    Inspired from PointAugmenting code
    https://github.com/VISION-SJTU/PointAugmenting/blob/main/det3d/datasets/utils/cross_modal_augmentation.py#L6

    Args:
        points: input 3d points array, shape (N, 3)

    Returns:
        points in spherical coordinates form, shape (N, 3)
        with last dimension being in the order (r, theta, phi)
        where r is radial distance, theta is angle from the polar (Z) axis,
        and phi is the angle from X axis on lateral X-Y plane.
    """
    pts_r = np.sqrt(
        np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2])
    )
    # Note: theta here is from the polar (Z) axis, and not from the X-Y lateral plane,
    # hence the arccos between z and r, i.e. arccos(z/r)
    pts_theta = np.arccos(points[:, 2] / pts_r)
    pts_phi = (
        np.arctan(points[:, 1] / points[:, 0]) + (points[:, 0] < 0) * np.pi + np.pi * 2
    ) % (
        np.pi * 2
    )  # [0, 2*pi]
    pts_rr = np.vstack([pts_r, pts_theta, pts_phi]).T
    return pts_rr


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """Add noise to every box (only on the horizontal plane).

    Args:
        boxes (np.ndarray): Input boxes with shape (N, 5).
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).

    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    """Add noise to every box (only on the horizontal plane). Version 2 used
    when enable global rotations.

    Args:
        boxes (np.ndarray): Input boxes with shape (N, 5).
        valid_mask (np.ndarray): Mask to indicate which boxes are valid
            with shape (N).
        loc_noises (np.ndarray): Location noises with shape (N, M, 3).
        rot_noises (np.ndarray): Rotation noises with shape (N, M).

    Returns:
        np.ndarray: Mask to indicate whether the noise is
            added successfully (pass the collision test).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                )
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask


def _select_transform(transform, indices):
    """Select transform.

    Args:
        transform (np.ndarray): Transforms to select from.
        indices (np.ndarray): Mask to indicate which transform to select.

    Returns:
        np.ndarray: Selected transforms.
    """
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """Get the 3D rotation matrix.

    Args:
        rot_mat_T (np.ndarray): Transposed rotation matrix.
        angle (float): Rotation angle.
        axis (int): Rotation axis.
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform, rot_transform, valid_mask):
    """Apply transforms to points and box centers.

    Args:
        points (np.ndarray): Input points.
        centers (np.ndarray): Input box centers.
        point_masks (np.ndarray): Mask to indicate which points need
            to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray): Mask to indicate which boxes are valid.
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = points[i : i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """Transform 3D boxes.

    Args:
        boxes (np.ndarray): 3D boxes to be transformed.
        loc_transform (np.ndarray): Location transform to be applied.
        rot_transform (np.ndarray): Rotation transform to be applied.
        valid_mask (np.ndarray | None): Mask to indicate which boxes are valid.
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def noise_per_object_v3_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=100,
):
    """Random rotate or remove each groundtruth independently. use kitti viewer
    to test this function points_transform_

    Args:
        gt_boxes (np.ndarray): Ground truth boxes with shape (N, 7).
        points (np.ndarray | None): Input point cloud with shape (M, 4).
            Default: None.
        valid_mask (np.ndarray | None): Mask to indicate which boxes are valid.
            Default: None.
        rotation_perturb (float): Rotation perturbation. Default: pi / 4.
        center_noise_std (float): Center noise standard deviation.
            Default: 1.0.
        global_random_rot_range (float): Global random rotation range.
            Default: pi/4.
        num_try (int): Number of try. Default: 100.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)

    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis], grot_uppers[..., np.newaxis], size=[num_boxes, num_try]
    )

    origin = (0.5, 0.5, 0)
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )

    # TODO: rewrite this noise box function?
    if not enable_grot:
        selected_noise = noise_per_box(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
        )
    else:
        selected_noise = noise_per_box_v2_(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises, global_rot_noises
        )

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        # TODO: replace this points_in_convex function by my tools?
        point_masks = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points, gt_boxes[:, :3], point_masks, loc_transforms, rot_transforms, valid_mask
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)
