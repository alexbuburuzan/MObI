import copy
import cv2
import numpy as np
import torch

from ldm.data.lidar_converter import LidarConverter
from torchvision.transforms import Resize
import torch.nn.functional as F


def resize(x, size, mode="avg_pool"):
    if mode == "avg_pool":
        _, _, height, width = x.shape
        x = F.avg_pool2d(x, kernel_size=(height // size[0], width // size[1]))
    elif mode == "max_pool":
        _, _, height, width = x.shape
        x = F.max_pool2d(x, kernel_size=(height // size[0], width // size[1]))
    elif mode == "nearest":
        x = F.interpolate(x, size=size, mode="nearest")
    else:
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return x


def get_image_coords(bbox_corners, lidar2image, include_depth=False):
    """
    Get the camera coordinates of the 3D bounding box

    Args:
        bbox_corners: np.array, shape (8, 3)
        lidar2image: np.array, shape (4, 4)
        include_depth: bool, whether to include the depth dimension

    Returns:
        np.array, shape (8, 3) if include_depth is True, else (8, 2)
        Each row is the x, y, (depth) coordinates of the 3D bounding box in the image frame
        x \in [0, W], y \in [0, H]
    """
    coords = np.concatenate(
        [bbox_corners.reshape(-1, 3), np.ones((8, 1))], axis=-1
    )
    lidar2image = lidar2image.copy().reshape(4, 4)
    coords = coords @ lidar2image.T
    coords = coords.reshape(8, 4)

    coords[..., 2] = np.clip(coords[..., 2], a_min=1e-5, a_max=1e5)
    coords[..., :2] /= coords[..., 2, None]

    if include_depth:
        coords = coords[..., :3].reshape(8, 3)
    else:
        coords = coords[..., :2].reshape(8, 2)

    return coords

def rotate_bbox(bbox_corners, angle=0):
    """
    Rotate the 3D bounding box around its y-axis

    Args:
        bbox_corners: np.array, shape (8, 3)
        angle: float, rotation angle in degrees

    Returns:
        np.array, shape (8, 3)
        Each row is the x, y, z coordinates
    """
    if angle == 0:
        return bbox_corners
    
    bbox_corners = copy.deepcopy(bbox_corners)
    angle = np.deg2rad(angle)
    center = np.mean(bbox_corners, axis=0)
    bbox_corners -= center

    rotation_matrix = np.array([
        [np.cos(angle),-np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    bbox_corners = bbox_corners @ rotation_matrix.T

    bbox_corners += center
    return bbox_corners


def translate_bbox(bbox_corners, new_center):
    """
    Translate the 3D bounding box to a new center

    Args:
        bbox_corners: np.array, shape (8, 3)
        new_center: np.array, shape (3,)

    Returns:
        np.array, shape (8, 3)
        Each row is the x, y, z coordinates
    """
    bbox_corners = copy.deepcopy(bbox_corners)
    center = np.mean(bbox_corners, axis=0)
    bbox_corners -= center
    bbox_corners += new_center
    return bbox_corners


def get_camera_coords(bbox_corners, lidar2camera):
    """
    Get the camera coordinates of the 3D bounding box

    Args:
        bbox_corners: np.array, shape (8, 3)
        lidar2camera: np.array, shape (4, 4)

    Returns:
        np.array, shape (8, 3)
        Each row is the x, y, z coordinates of the 3D bounding box in the camera frame
    """
    coords = np.concatenate(
        [bbox_corners.reshape(-1, 3), np.ones((8, 1))], axis=-1
    )
    lidar2camera = lidar2camera.copy().reshape(4, 4)
    coords = coords @ lidar2camera.T
    coords = coords.reshape(8, 4)

    return coords[..., :3]

def get_inpaint_mask(bbox_corners, transform, H, W, expand_ratio=0.1, use_3d_edit_mask=True):
    if use_3d_edit_mask:
        bbox_corners = expand_bbox_corners(bbox_corners, expand_ratio)
        mask = np.zeros((H, W), dtype=np.uint8)

        coords = get_image_coords(bbox_corners, transform)

        # Draw 3D boxes
        for polygon in [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 4, 7, 3],
            [1, 5, 6, 2],
        ]:
            points = coords[polygon].astype(np.int32)
            cv2.fillPoly(mask, [points], 1, cv2.LINE_AA)
    else:
        bbox_2d = get_2d_bbox(bbox_corners, transform, H, W, expand_ratio)
        mask = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = bbox_2d
        mask[y1:y2, x1:x2] = 1

    mask = 1. - torch.tensor(mask > 0.5).float()
    return mask


def get_range_inpaint_mask(bbox_corners, range_height, range_width, expand_ratio=0.1, crop_left=None):
    bbox_corners = expand_bbox_corners(bbox_corners, expand_ratio)
    mask = np.zeros((range_height, range_width), dtype=np.uint8)

    lidar_converter = LidarConverter()
    coords = lidar_converter.get_range_coords(bbox_corners)
    _, _, _, coords, _, _ = lidar_converter.apply_default_transforms(
        coords, height=range_height, width=range_width, crop_left=crop_left
    )
    coords = coords[:, :2]

    for polygon in [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 4, 7, 3],
        [1, 5, 6, 2],
    ]:
        points = coords[polygon].astype(np.int32)
        cv2.fillPoly(mask, [points], 1, cv2.LINE_AA)

    mask = 1. - torch.tensor(mask > 0.5).float()
    return mask


def draw_projected_bbox(image, bbox_coords, color=(0, 165, 255), thickness=2):
    """
    Draw projected 3D bounding box on the image

    Args:
        image: np.array, shape (H, W, 3)
        bbox_coords: np.array, shape (8, 2)
        color: tuple, color of the bbox
        thickness: int, thickness of the lines

    Returns:
        np.array, shape (H, W, 3)
    """
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))

    H, W = image.shape[:2]
    bbox_coords = bbox_coords.copy()
    bbox_coords[..., 0] *= W
    bbox_coords[..., 1] *= H
    bbox_coords = bbox_coords.astype(np.int32)

    canvas = image.copy()

    for start, end in [
        (0, 1), (0, 3), (3, 2), (1, 2), # bottom lines
        (1, 5), (0, 4), (3, 7), (2, 6), # vertical lines
        (4, 7), (4, 5), (5, 6), (6, 7), # top lines
    ]:
        cv2.line(
            canvas,
            bbox_coords[start].astype(np.int32),
            bbox_coords[end].astype(np.int32),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Draw arrow towards the face 0 1 4 5
    center = np.mean(bbox_coords, axis=0).astype(int)
    tip = np.mean(bbox_coords[[0, 1, 4, 5]], axis=0).astype(int)
    cv2.arrowedLine(
        canvas,
        center,
        tip,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.1,
    )

    return canvas


def get_2d_bbox(bbox_corners, transform, H, W, expand_ratio=0.1):
    bbox_corners = expand_bbox_corners(bbox_corners, expand_ratio)
    coords = get_image_coords(bbox_corners, transform)

    minxy = np.min(coords, axis=-2)
    maxxy = np.max(coords, axis=-2)

    bbox_2d = np.concatenate([minxy, maxxy], axis=-1).astype(int)
    bbox_2d[0::2] = np.clip(bbox_2d[0::2], a_min=0, a_max=W - 1)
    bbox_2d[1::2] = np.clip(bbox_2d[1::2], a_min=0, a_max=H - 1)

    return bbox_2d


def expand_bbox_corners(bbox_corners, expand_ratio=0.1):
    if expand_ratio == 0:
        return bbox_corners

    bbox_corners = copy.deepcopy(bbox_corners)
    center = np.mean(bbox_corners, axis=0)
    bbox_corners -= center
    bbox_corners *= (1 + expand_ratio)
    bbox_corners += center

    return bbox_corners

def visualize_lidar(
    lidar=None,
    *,
    fpath=None,
    bboxes=None,
    xlim=(-10, 10),
    ylim=(-10, 10),
    thickness: int = 1,
    bbox_color=(0, 165, 255),
    points_color=(0, 128, 128),
    dpi: int = 20,  # Set to desired resolution
) -> np.ndarray:
    bbox = copy.deepcopy(bboxes)
    lidar = lidar.copy() if lidar is not None else None

    # Create a blank image
    img = np.ones((int((ylim[1]-ylim[0])*dpi), int((xlim[1]-xlim[0])*dpi), 3), dtype=np.uint8) * 255

    if bboxes is not None and len(bboxes) > 0:
        if bboxes.ndim == 2:
            bboxes = bboxes[None, ...]
        for bbox in bboxes:
            for start, end in [
                (0, 1), (0, 3), (3, 2), (1, 2),  # bottom lines
                (1, 5), (0, 4), (3, 7), (2, 6),  # vertical lines
                (4, 7), (4, 5), (5, 6), (6, 7),  # top lines
            ]:
                pt1 = (int(bbox[start, 0]*dpi-xlim[0]*dpi), int((ylim[1]-bbox[start, 1])*dpi))
                pt2 = (int(bbox[end, 0]*dpi-xlim[0]*dpi), int((ylim[1]-bbox[end, 1])*dpi))
                cv2.line(img, pt1, pt2, bbox_color, thickness)

            # draw an arrow for the orientation
            center = np.mean(bbox, axis=0)
            tip = np.mean(bbox[[0, 1, 4, 5]], axis=0)
            pt_center = (int(center[0]*dpi-xlim[0]*dpi), int((ylim[1]-center[1])*dpi))
            pt_tip = (int(tip[0]*dpi-xlim[0]*dpi), int((ylim[1]-tip[1])*dpi))
            cv2.arrowedLine(
                img,
                pt_center,
                pt_tip,
                bbox_color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.1,
            )

    if lidar is not None:
        lidar[:, 0] = (lidar[:, 0] - xlim[0]) * dpi 
        lidar[:, 1] = (ylim[1] - lidar[:, 1]) * dpi
        mask = (lidar[:, 0] >= 0) & (lidar[:, 0] < img.shape[1]) & (lidar[:, 1] >= 0) & (lidar[:, 1] < img.shape[0])
        lidar = lidar[mask].astype(int)

        img[lidar[:, 1], lidar[:, 0]] = points_color

    if fpath is not None:
        cv2.imwrite(fpath, img[..., ::-1])    

    return img


def focus_on_bbox(points, bbox_3d):
    points = points.copy()
    bbox_3d = bbox_3d.copy()

    bbox_center = np.mean(bbox_3d, axis=0)

    sign = 1 if bbox_center[0] > 0 else -1
    theta_z = sign * np.pi / 4
    rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    theta_x = -np.pi / 3
    rot_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_mat = np.dot(rot_x, rot_z)

    points = points - bbox_center
    points = np.dot(points, rot_mat.T)

    bbox_3d = bbox_3d - bbox_center
    bbox_3d = np.dot(bbox_3d, rot_mat.T)

    return points, bbox_3d


def un_norm(x, size=(256, 256)):
    return (Resize(size)(x) + 1.0)/2.0


def un_norm_clip(x, size=(256, 256)):
    x = Resize(size)(x)
    x[:,0] = x[:,0] * 0.26862954 + 0.48145466
    x[:,1] = x[:,1] * 0.26130258 + 0.4578275
    x[:,2] = x[:,2] * 0.27577711 + 0.40821073
    return x


def get_images(x, transform=None, bboxes=None):
    x = transform(x) if transform is not None else x
    x = x.cpu().numpy().transpose(0, 2, 3, 1)
    x = (x * 255).astype(np.uint8)
    if bboxes is not None:
        x = np.stack(
            [draw_projected_bbox(
                x,
                bboxes[i, :, :2],
                color=(255, 165, 0),
                thickness=1,
            ) for i, x in enumerate(x)]
        )
    return torch.from_numpy(x).permute(0, 3, 1, 2)


def get_camera_vis(
    sample,
    input,
    inpaint_input,
    reference,
    rec,
    ref_bboxes=None,
):
    ref_bboxes = ref_bboxes.cpu().numpy()
    sample = get_images(sample, transform=un_norm, bboxes=ref_bboxes)
    input = get_images(input, transform=un_norm, bboxes=ref_bboxes)
    inpaint_input = get_images(inpaint_input, transform=un_norm, bboxes=ref_bboxes)
    reference = get_images(reference, transform=un_norm_clip)
    rec = get_images(rec, transform=un_norm, bboxes=ref_bboxes)

    return sample, input, inpaint_input, reference, rec


def get_lidar_vis(
    *,
    sample,
    input,
    rec,
    bboxes,
    range_depth_orig,
    range_shift_left,
    range_pitch,
    range_yaw,
    width_crop,
):
    sample_vis, input_vis, rec_vis = [], [], []
    bboxes = bboxes.cpu().numpy()
    range_pitch = range_pitch.cpu().numpy()
    range_yaw = range_yaw.cpu().numpy()

    sample = postprocess_range_depth(
        range_depth=sample,
        range_depth_orig=range_depth_orig,
        crop_left=range_shift_left,
        width_crop=width_crop,
        zero_context=True
    )
    input = postprocess_range_depth(
        range_depth=input,
        range_depth_orig=range_depth_orig,
        crop_left=range_shift_left,
        width_crop=width_crop,
        zero_context=True
    )
    rec = postprocess_range_depth(
        range_depth=rec,
        range_depth_orig=range_depth_orig,
        crop_left=range_shift_left,
        width_crop=width_crop,
        zero_context=True
    )
    lidar_converter = LidarConverter()

    for i in range(len(sample)):
        bbox_3d = bboxes[i]

        sample_pc, _ = lidar_converter.range2pcd(sample[i], range_pitch[i], range_yaw[i])
        input_pc, _ = lidar_converter.range2pcd(input[i], range_pitch[i], range_yaw[i])
        rec_pc, _ = lidar_converter.range2pcd(rec[i], range_pitch[i], range_yaw[i])

        sample_pc, _ = focus_on_bbox(sample_pc, bbox_3d)
        input_pc, _ = focus_on_bbox(input_pc, bbox_3d)
        rec_pc, bbox_3d = focus_on_bbox(rec_pc, bbox_3d)

        sample_vis.append(visualize_lidar(sample_pc, bboxes=bbox_3d, bbox_color=(255, 165, 0)))
        input_vis.append(visualize_lidar(input_pc, bboxes=bbox_3d, bbox_color=(255, 165, 0)))
        rec_vis.append(visualize_lidar(rec_pc, bboxes=bbox_3d, bbox_color=(255, 165, 0)))

    sample_vis = torch.from_numpy(np.stack(sample_vis)).permute(0, 3, 1, 2)
    input_vis = torch.from_numpy(np.stack(input_vis)).permute(0, 3, 1, 2)
    rec_vis = torch.from_numpy(np.stack(rec_vis)).permute(0, 3, 1, 2)

    return sample_vis, input_vis, rec_vis


def postprocess_range_depth_int(
    *,
    range_depth,
    range_depth_orig,
    range_int,
    range_int_orig,
    crop_left,
    width_crop,
    zero_context=False
):
    range_depth = range_depth.cpu().numpy()
    range_depth_orig = range_depth_orig.cpu().numpy()
    range_int = range_int.cpu().numpy()
    range_int_orig = range_int_orig.cpu().numpy()

    if zero_context:
        range_depth_orig = range_depth_orig * 0 - 1
    
    lidar_converter = LidarConverter()

    range_depth_final_all, range_depth_int_all = [], []
    for i in range(len(range_depth)):
        range_depth_final, range_int_final = lidar_converter.undo_default_transforms(
            crop_left=crop_left[i].item(),
            width_crop=width_crop[i].item(),
            range_depth_crop=range_depth[i, 0],
            range_depth=range_depth_orig[i],
            range_int_crop=range_int[i, 0],
            range_int=range_int_orig[i],
        )

        range_depth_final_all.append(range_depth_final)
        range_depth_int_all.append(range_int_final)

    return np.stack(range_depth_final_all), np.stack(range_depth_int_all)

def postprocess_range_depth(
    *,
    range_depth,
    range_depth_orig,
    crop_left,
    width_crop,
    zero_context=False
):
    range_depth = range_depth.cpu().numpy()
    range_depth_orig = range_depth_orig.cpu().numpy()

    if zero_context:
        range_depth_orig = range_depth_orig * 0 - 1
    
    lidar_converter = LidarConverter()

    range_depth_final = []
    for i in range(len(range_depth)):
        range_depth_final.append(
            lidar_converter.undo_default_transforms(
                crop_left=crop_left[i].item(),
                width_crop=width_crop[i].item(),
                range_depth_crop=range_depth[i, 0],
                range_depth=range_depth_orig[i],
            )[0]
        )

    return np.stack(range_depth_final)


def depth_normalization(depth, min_d, max_d, alpha=0.75):
    assert -1 <= min_d < max_d <= 1, "min_d and max_d must be in the range -1 to 1 and min_d < max_d"
    assert 0 < alpha <= 1, "alpha must be in the range 0 to 1"
    
    # Create a tensor to store the normalized depth values
    normalized_depth = torch.empty_like(depth)
    min_d, max_d = min_d.to(depth.dtype), max_d.to(depth.dtype)
    
    # Normalize values between min_d and max_d to [-alpha, alpha]
    mask_mid = (depth >= min_d) & (depth <= max_d)
    normalized_depth[mask_mid] = -alpha + 2 * alpha * (depth[mask_mid] - min_d) / (max_d - min_d)
    
    # Normalize values between -1 and min_d to [-1, -alpha]
    mask_low = (depth >= -1) & (depth < min_d)
    normalized_depth[mask_low] = -1 + -(alpha - 1) * (depth[mask_low] + 1) / (min_d + 1)
    
    # Normalize values between max_d and 1 to [alpha, 1]
    mask_high = (depth > max_d) & (depth <= 1)
    normalized_depth[mask_high] = alpha + (1 - alpha) * (depth[mask_high] - max_d) / (1 - max_d)
    
    return normalized_depth


def inverse_depth_normalization(normalized_depth, min_d, max_d, alpha=0.75):
    assert -1 <= min_d < max_d <= 1, "min_d and max_d must be in the range -1 to 1 and min_d < max_d"
    assert 0 < alpha <= 1, "alpha must be in the range 0 to 1"
    
    # Create a tensor to store the original depth values
    depth = torch.empty_like(normalized_depth)
    min_d, max_d = min_d.to(depth.dtype), max_d.to(depth.dtype)
    
    # Inverse normalization for values between -alpha and alpha to [min_d, max_d]
    mask_mid = (normalized_depth >= -alpha) & (normalized_depth <= alpha)
    depth[mask_mid] = min_d + (normalized_depth[mask_mid] + alpha) * (max_d - min_d) / (2 * alpha)
    
    # Inverse normalization for values between -1 and -alpha to [-1, min_d]
    mask_low = (normalized_depth >= -1) & (normalized_depth < -alpha)
    depth[mask_low] = -1 + -(normalized_depth[mask_low] + 1) * (min_d + 1) / (alpha - 1)
    
    # Inverse normalization for values between alpha and 1 to [max_d, 1]
    mask_high = (normalized_depth > alpha) & (normalized_depth <= 1)
    depth[mask_high] = max_d + (normalized_depth[mask_high] - alpha) * (1 - max_d) / (1 - alpha)
    
    return depth