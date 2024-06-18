import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def visualize_camera(
    image: np.ndarray,
    *,
    fpath: str = None,
    bboxes: LiDARInstance3DBoxes = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
    radius: int = 2,
    save_figure: bool = False,
    show_image: bool = True,
    filled: bool = False,
) -> np.ndarray:
    """
    Args:
        image (np.ndarray): Image to be visualized. 
            Shape: (H, W, 3). Range: [0, 255]. Dtype: uint8.
        fpath (str): Path to save the figure.
        bboxes (LiDARInstance3DBoxes): 3D boxes.
        points (np.ndarray): Lidar points.
        labels (np.ndarray): Labels of the boxes.
        transform (np.ndarray): Transform matrix from lidar to image.
        classes (list[str]): Class names of the boxes.
        color (tuple[int]): Color of the boxes.
        thickness (float): Thickness of the boxes.
        save_figure (bool): Whether to save the figure.
        show_image (bool): Whether to show the image. If False, the canvas will be black.
        filled (bool): Whether to fill the 3d projected boxes.

    Returns:
        np.ndarray: Image with drawn bboxes.
    """
    transform = transform.copy()
    bboxes = copy.deepcopy(bboxes)

    if show_image:
        canvas = image.copy()
    else:
        canvas = np.zeros_like(image) + 255
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Plot lidar points
    if points is not None and len(points) > 0:
        points = points[..., :3].copy()
        points = np.concatenate(
            [points, np.ones((points.shape[0], 1))], axis=-1
        )
        distance = np.linalg.norm(points, axis=-1)
        # print(distance.min())
        # distance = distance - distance.min()
        # distance = np.log(distance + 1e-5)
        distance = np.clip(distance, np.percentile(distance, 5), np.percentile(distance, 95))
        distance = 1 - (distance - distance.min()) / (distance.max() - distance.min())

        transform = transform.reshape(4, 4)
        points = points @ transform.T
        points = points.reshape(-1, 4)

        index = points[:, 2] > 0
        points = points[index]
        distance = distance[index]

        points = points.reshape(-1, 4)
        points[:, 2] = np.clip(points[:, 2], a_min=1e-5, a_max=1e5)
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]
        points = points[..., :2]

        index = (points[:, 0] > 1) & (points[:, 1] > 1) & (points[:, 0] < canvas.shape[1] - 1) & (points[:, 1] < canvas.shape[0] - 1)
        points = points[index]
        distance = distance[index]
        points = points.astype(np.int)

        colours = plt.cm.jet(1 - distance)[:, :3] * 255
        canvas[points[:, 1]    , points[:, 0]    ] = colours
        canvas[points[:, 1] + 1, points[:, 0] + 1] = colours
        canvas[points[:, 1]    , points[:, 0] + 1] = colours
        canvas[points[:, 1] + 1, points[:, 0]    ] = colours

    # Plot 3D boxes
    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords[..., 2] = np.clip(coords[..., 2], a_min=1e-5, a_max=1e5)
        coords[..., :2] /= coords[..., 2, None]

        coords = coords[..., :2].reshape(-1, 8, 2)

        if filled:
            for index in range(coords.shape[0]):
                for polygon in [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [2, 3, 7, 6],
                    [0, 4, 7, 3],
                    [1, 5, 6, 2],
                ]:
                    points = coords[index, polygon].astype(np.int)
                    cv2.fillPoly(
                        canvas,
                        [points],
                        color or OBJECT_PALETTE[classes[labels[index]]],
                        cv2.LINE_AA,
                    )
        else:
            for index in range(coords.shape[0]):
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        canvas,
                        coords[index, start].astype(np.int),
                        coords[index, end].astype(np.int),
                        color or OBJECT_PALETTE[classes[labels[index]]],
                        thickness,
                        cv2.LINE_AA,
                    )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    if save_figure:
        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        mmcv.imwrite(canvas, fpath)
    
    return canvas


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


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
