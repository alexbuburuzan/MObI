import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ldm.data.lidar_converter import LidarConverter

def get_image_coords(bbox_corners, lidar2image):
    """
    Get the camera coordinates of the 3D bounding box

    Args:
        bbox_corners: np.array, shape (8, 3)
        lidar2image: np.array, shape (4, 4)

    Returns:
        np.array, shape (8, 2)
        Each row is the x, y coordinates of the 3D bounding box in the image
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

def get_inpaint_mask(bbox_corners, transform, H, W, expand_ratio=0.1, use_3d_edit_mask=True, use_lidar=False, crop_left=None):
    if use_3d_edit_mask:
        bbox_corners = expand_bbox_corners(bbox_corners, expand_ratio)
        mask = np.zeros((H, W), dtype=np.uint8)

        if not use_lidar:
            coords = get_image_coords(bbox_corners, transform)
        else:
            lidar_converter = LidarConverter()
            coords = lidar_converter.get_range_coords(bbox_corners)
            _, _, coords, _ = lidar_converter.apply_default_transforms(
                coords, image_height=H, image_width=W, crop_left=crop_left
            )
            coords = coords[:, :2]

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

    mask = ((mask > 0.5) * 255).astype(np.uint8)
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
    color=(0, 165, 255),
    dpi: int = 40,  # Set to desired resolution
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
                cv2.line(img, pt1, pt2, color, thickness)

            # draw an arrow for the orientation
            center = np.mean(bbox, axis=0)
            tip = np.mean(bbox[[0, 1, 4, 5]], axis=0)
            pt_center = (int(center[0]*dpi-xlim[0]*dpi), int((ylim[1]-center[1])*dpi))
            pt_tip = (int(tip[0]*dpi-xlim[0]*dpi), int((ylim[1]-tip[1])*dpi))
            cv2.arrowedLine(
                img,
                pt_center,
                pt_tip,
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.1,
            )

    if lidar is not None:
        lidar[:, 0] = (lidar[:, 0] - xlim[0]) * dpi
        lidar[:, 1] = (ylim[1] - lidar[:, 1]) * dpi
        mask = (lidar[:, 0] >= 0) & (lidar[:, 0] < img.shape[1]) & (lidar[:, 1] >= 0) & (lidar[:, 1] < img.shape[0])
        lidar = lidar[mask].astype(int)

        img[lidar[:, 1], lidar[:, 0]] = (0, 128, 128)    

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