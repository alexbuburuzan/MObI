from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Tuple, Union, Optional

import os
import cv2
import copy
import pickle
import random
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data

import albumentations as A

class LidarConverter:
    def __init__(
        self,
        H=32,
        W=1024,
        fov=(10, -30),
        depth_range=(1, 51.2),
        log_scale=True,
        depth_scale=5.7,
    ) -> None:
        self.H = H
        self.W = W
        self.fov = fov
        self.depth_range = depth_range
        self.fov_up = fov[0] / 180.0 * np.pi
        self.fov_down = fov[1] / 180.0 * np.pi
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)
        self.size = (H, W)
        self.log_scale = log_scale
        self.depth_scale = depth_scale

    def pcd2range(self, pcd, labels=None):
        pcd = pcd.copy()
        if labels is not None:
            labels = labels.copy()
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
        proj_x *= self.W # in [0.0, W]
        proj_y *= self.H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.maximum(0, np.minimum(self.size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
        proj_y = np.maximum(0, np.minimum(self.size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        proj_x, proj_y = proj_x[order], proj_y[order]

        # project depth
        depth = depth[order]
        proj_range = np.full(self.size, -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        # project point feature
        if labels is not None:
            labels = labels[mask][order]
            proj_feature = np.full(self.size, 0, dtype=np.float32)
            proj_feature[proj_y, proj_x] = labels
        else:
            proj_feature = None

        proj_range = np.where(proj_range < 0, 0, proj_range)
        if self.log_scale:
            proj_range = np.log2(proj_range + 0.0001 + 1)

            proj_range = proj_range / self.depth_scale
            proj_range = proj_range * 2. - 1.

            proj_range = np.clip(proj_range, -1, 1)

        return proj_range, proj_feature, mask

    def range2pcd(self, depth, label=None):
        depth = depth.copy()
        if label is not None:
            label = label.copy()

        if self.log_scale:
            depth = (depth + 1) / 2
            depth = (depth * self.depth_scale)
            depth = np.exp2(depth) - 1

        depth = depth.flatten()

        scan_x, scan_y = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
        scan_x = scan_x.astype(np.float32) / self.size[1]
        scan_y = scan_y.astype(np.float32) / self.size[0]

        yaw = (np.pi * (scan_x * 2 - 1)).flatten()
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down)).flatten()

        pcd = np.zeros((len(yaw), 3)).astype(np.float32)
        pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
        pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
        pcd[:, 2] = np.sin(pitch) * depth

        # mask out invalid points
        mask = np.logical_and(depth > self.depth_range[0], depth < self.depth_range[1])
        pcd = pcd[mask, :]

        # label
        if label is not None:
            label = label.flatten()[mask]

        return pcd, label
    
    def resize(self, range_img, feature=None, new_W=1034, new_H=32):
        range_img = cv2.resize(range_img, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        if feature is not None:
            feature = cv2.resize(feature, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        return range_img, feature

    def get_range_coords(self, bbox_corners):
        bbox_corners = bbox_corners.copy()

        # get depth (distance) of all points
        depth = np.linalg.norm(bbox_corners, 2, axis=1)

        # get scan components
        scan_x, scan_y, scan_z = bbox_corners[:, 0], bbox_corners[:, 1], bbox_corners[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_range  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.W # in [0.0, W]
        proj_y *= self.H  # in [0.0, H]

        coords = np.concatenate(
            [proj_x[:, None], proj_y[:, None]],
            axis=-1
        ).astype(np.int32)

        return coords


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

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


class NuScenesDataset(data.Dataset):
    def __init__(
        self,
        state,
        object_database_path,
        scene_database_path,
        object_classes,
        expand_mask_ratio=0,
        expand_ref_ratio=0,
        ref_aug=True,
        prob_use_3d_edit_mask=1,
        ref_mode="same-ref", # same-ref, track-ref, random-ref, no-ref
        image_height=512,
        image_width=512,
        reference_image_min_h=40,
        reference_image_min_w=40,
        frustum_iou_max=0.7,
        camera_visibility_min=0.5,
        rot_every_angle=0,
        specific_scene=None, # used for rotation test
        use_lidar=False,
    ) -> None:
        self.state = state
        self.ref_aug = ref_aug
        self.ref_mode = ref_mode
        self.expand_mask_ratio = expand_mask_ratio
        self.expand_ref_ratio = expand_ref_ratio
        self.specific_scene = specific_scene
        self.prob_use_3d_edit_mask = prob_use_3d_edit_mask
        self.use_lidar = use_lidar
        self.image_height = image_height
        self.image_width = image_width

        self.all_objects_meta = pd.read_csv(object_database_path, index_col=0)
        # filter out small, occluded objects
        self.all_objects_meta = self.all_objects_meta[
            (self.all_objects_meta["reference_image_h"] >= reference_image_min_w) &
            (self.all_objects_meta["reference_image_w"] >= reference_image_min_h) &
            (self.all_objects_meta["max_iou_overlap"] <= frustum_iou_max) &
            self.all_objects_meta["object_class"].isin(object_classes) &
            (self.all_objects_meta["camera_visibility_mask"] >= camera_visibility_min)
        ]

        if self.state == "test":
            # select an object from each class
            self.objects_meta = self.all_objects_meta.groupby("object_class").apply(
                lambda x: x.sample(1)
            ).reset_index(drop=True)
        else:
            self.objects_meta = self.all_objects_meta

        if rot_every_angle != 0:
            angles = np.arange(0, 360, rot_every_angle)
            self.objects_meta = pd.concat(
                [self.objects_meta] * len(angles), ignore_index=True
            )
            self.objects_meta["bbox_rot_angle"] = np.tile(angles, len(self.objects_meta) // len(angles))

        with open(scene_database_path, "rb") as f:
            self.scenes_info = pickle.load(f)

        # Image transforms
        ref_augs = [
            A.Resize(height=224, width=224)
        ]
        if ref_aug:
            ref_augs += [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20),
                A.Blur(p=0.3),
                A.ElasticTransform(p=0.3)
            ]
        self.ref_transform = A.Compose(ref_augs)
        self.resize = T.Resize([image_height, image_width])

    def __getitem__(self, index):
        object_meta = self.objects_meta.iloc[index]

        if self.specific_scene is not None:
            scene_info = self.scenes_info[self.specific_scene]
            # always use the front camera when specific_scene is provided
            cam_idx = 0
        else:
            scene_info = self.scenes_info[object_meta["scene_token"]]
            cam_idx = object_meta["cam_idx"]

        id_name = self.get_id_name(object_meta)
        
        lidar2image = scene_info["lidar2image_transforms"][cam_idx]
        lidar2camera = scene_info["lidar2camera_transforms"][cam_idx]
        image_path = scene_info["image_paths"][cam_idx]
        bbox_3d = scene_info["gt_bboxes_3d_corners"][object_meta["scene_obj_idx"]]

        # Range image
        range_image_crop, range_int_crop, bbox_range_coords = self.get_rasterized_lidar(scene_info, bbox_3d)

        # Image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        image_tensor = get_tensor()(np.array(image))
        image_tensor = self.resize(image_tensor)

        if self.use_lidar:
            image_tensor = range_image_crop

        # Reference
        ref_image, ref_bbox_3d, ref_label = self.get_reference(object_meta)
        if self.specific_scene is not None:
            bbox_3d = ref_bbox_3d

        ref_image = self.ref_transform(image=ref_image)["image"]
        ref_image = Image.fromarray(ref_image)
        ref_image_tensor = get_tensor_clip()(ref_image)

        bbox_rot_angle = object_meta.get("bbox_rot_angle", 0)
        id_name += "_rot-{}".format(bbox_rot_angle)
        bbox_3d = rotate_bbox(bbox_3d, bbox_rot_angle)

        if self.specific_scene is not None:
            bbox_3d = translate_bbox(bbox_3d, [0, 9, -1])
       
        bbox_image_coords = get_image_coords(bbox_3d, lidar2image)
        bbox_image_coords[..., 0] /= W
        bbox_image_coords[..., 1] /= H
        bbox_camera_coords = get_camera_coords(bbox_3d, lidar2camera)

        if self.use_lidar:
            bbox_image_coords = bbox_range_coords

        # Mask
        use_3d_edit_mask = (random.random() < self.prob_use_3d_edit_mask)
        mask_np = get_inpaint_mask(
            bbox_3d, lidar2image, H, W, self.expand_mask_ratio, use_3d_edit_mask
        )
        mask_image = Image.fromarray(mask_np)
        mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_image)
        mask_tensor = (self.resize(mask_tensor) > 0.5).float()

        # Inpainted image
        inpaint_tensor = image_tensor * mask_tensor

        data = {
            "id_name": id_name,
            "GT": image_tensor,
            "inpaint_image": inpaint_tensor,
            "inpaint_mask": mask_tensor,
            "bbox_image_coords": bbox_image_coords,
            "cond": {
                "ref_image": ref_image_tensor,
                "ref_bbox": bbox_camera_coords,
                "ref_label": ref_label,
            }
        }

        return data
    
    def __len__(self):
        return len(self.objects_meta)
    
    def get_reference(self, current_object_meta):
        if self.ref_mode == "no-ref":
            return np.zeros((224, 224, 3), dtype=np.uint8), None, 0
        elif self.ref_mode == "same-ref":
            reference_meta = current_object_meta
        elif self.ref_mode == "random-ref":
            reference_meta = self.all_objects_meta[
                self.all_objects_meta["object_class"] == current_object_meta["object_class"]
            ].sample(1).iloc[0]
        elif self.ref_mode == "track-ref":
            reference_meta = self.all_objects_meta[
                self.all_objects_meta["track_id"] == current_object_meta["track_id"]
            ].sample(1).iloc[0]
        else:
            raise ValueError("Invalid ref_mode")

        ref_obj_idx = reference_meta["scene_obj_idx"]
        cam_idx = reference_meta["cam_idx"]
        ref_scene_info = self.scenes_info[reference_meta["scene_token"]]
        lidar2image = ref_scene_info["lidar2image_transforms"][cam_idx]
        image_path = ref_scene_info["image_paths"][cam_idx]

        ref_bbox_3d = ref_scene_info["gt_bboxes_3d_corners"][ref_obj_idx]
        ref_label = ref_scene_info["gt_labels"][ref_obj_idx]
        
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        image_np = np.array(image)

        bbox_2d = get_2d_bbox(
            ref_bbox_3d, lidar2image, H, W, self.expand_ref_ratio
        )
        x1, y1, x2, y2 = bbox_2d
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)
        ref_image = image_np[y1:y1+h, x1:x1+w]

        return ref_image, ref_bbox_3d, ref_label
    
    def get_id_name(self, object_meta):
        id_name = "sample-{}_track-{}_time-{}_{}_{}".format(
            object_meta["scene_token"],
            object_meta["track_id"],
            object_meta["timestamp"],
            object_meta["object_class"],
            self.ref_mode
        )
        if self.ref_aug:
            id_name += "-aug"

        return id_name
    
    def get_rasterized_lidar(self, scene_info, bbox_3d):
        lidar_converter = LidarConverter()

        if "range_depth_path" in scene_info and "range_intensities_path" in scene_info:
            range_depth = np.load(scene_info["range_depth_path"])
            range_int = np.load(scene_info["range_intensity_path"])
        elif "lidar_path" in scene_info:
            lidar_scan = np.load(scene_info["lidar_path"])
            points = lidar_scan[:, :3].astype(np.float32)
            range_depth, range_int = lidar_converter.points2range(points, labels=lidar_scan[:, 3])
        else:
            raise ValueError("No lidar data found")

        bbox_range_coords = lidar_converter.get_range_coords(bbox_3d)
        bbox_range_coords[:, 0] += range_depth.shape[1]
        range_depth_ext = np.tile(range_depth, 3)
        range_int_ext = np.tile(range_int, 3)

        center_x = int(np.mean(bbox_range_coords[:, 0]))

        d_left = random.randint(self.image_width // 4, self.image_width - self.image_width // 4)
        d_right = self.image_width - d_left
        range_depth_crop = range_depth_ext[:, center_x - d_left: center_x + d_right]
        range_int_crop = range_int_ext[:, center_x - d_left: center_x + d_right]
        bbox_range_coords = bbox_range_coords - np.array([center_x - d_left, 0])

        bbox_range_coords = bbox_range_coords.astype(np.float32)
        bbox_range_coords[..., 0] /= range_depth_crop.shape[1]
        bbox_range_coords[..., 1] /= range_depth_crop.shape[0]

        range_depth_crop = np.tile(range_depth_crop[:, :, None], 3)
        range_depth_crop = get_tensor(normalize=False, toTensor=True)(range_depth_crop)
        range_depth_crop = self.resize(range_depth_crop)

        return range_depth_crop, range_int_crop, bbox_range_coords
        
        
