import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image

import torchvision.transforms as T
import torch.utils.data as data

import torchvision
import albumentations as A

from ldm.data.utils import (
    get_image_coords,
    rotate_bbox,
    translate_bbox,
    get_2d_bbox,
    get_camera_coords,
    get_inpaint_mask,
)
from ldm.data.lidar_converter import LidarConverter

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
        rot_test_scene=None, # used for rotation test
        use_lidar=False,
    ) -> None:
        self.state = state
        self.ref_aug = ref_aug
        self.ref_mode = ref_mode
        self.expand_mask_ratio = expand_mask_ratio
        self.expand_ref_ratio = expand_ref_ratio
        self.rot_test_scene = rot_test_scene
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

        if self.rot_test_scene is not None:
            scene_info = self.scenes_info[self.rot_test_scene]
            # always use the front camera when rot_test_scene is provided
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
        image_depth, image_int, bbox_range_coords, crop_left, range_depth, range_int = self.get_rasterized_lidar(scene_info, bbox_3d)

        # Image
        if not self.use_lidar:
            image = Image.open(image_path).convert("RGB")
            W, H = image.size
            image_tensor = get_tensor()(np.array(image))
            image_tensor = self.resize(image_tensor)

        if self.use_lidar:
            image_tensor = image_depth
            W, H = image_tensor.shape[2], image_tensor.shape[1]

        # Reference
        ref_image, ref_bbox_3d, ref_label = self.get_reference(object_meta)
        if self.rot_test_scene is not None:
            bbox_3d = ref_bbox_3d

        ref_image = self.ref_transform(image=ref_image)["image"]
        ref_image = Image.fromarray(ref_image)
        ref_image_tensor = get_tensor_clip()(ref_image)

        bbox_rot_angle = object_meta.get("bbox_rot_angle", 0)
        bbox_3d = rotate_bbox(bbox_3d, bbox_rot_angle)

        if self.rot_test_scene is not None:
            bbox_3d = translate_bbox(bbox_3d, [0, 9, -1])

        if not self.use_lidar:
            bbox_image_coords = get_image_coords(bbox_3d, lidar2image)
            bbox_image_coords[..., 0] /= W
            bbox_image_coords[..., 1] /= H
            bbox_cond_coords = get_camera_coords(bbox_3d, lidar2camera)
        if self.use_lidar:
            bbox_image_coords = bbox_range_coords
            bbox_cond_coords = bbox_range_coords

        # Mask
        use_3d_edit_mask = (random.random() < self.prob_use_3d_edit_mask)
        mask_np = get_inpaint_mask(
            bbox_3d, lidar2image, H, W, self.expand_mask_ratio, use_3d_edit_mask, self.use_lidar, crop_left
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
            "orig_range_depth": range_depth,
            "orig_range_int": range_int,
            "crop_left": crop_left,
            "bbox_3d": bbox_3d,
            "cond": {
                "ref_image": ref_image_tensor,
                "ref_bbox": bbox_cond_coords,
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
        id_name = "sample-{}_track-{}_time-{}_{}_{}_rot-{}".format(
            object_meta["scene_token"],
            object_meta["track_id"],
            object_meta["timestamp"],
            object_meta["object_class"],
            self.ref_mode,
            object_meta.get("bbox_rot_angle", 0),
        )
        if self.ref_aug:
            id_name += "-aug"

        return id_name
    
    def get_rasterized_lidar(self, scene_info, bbox_3d):
        lidar_converter = LidarConverter()

        if "range_depth_path" in scene_info and "range_intensity_path" in scene_info:
            range_depth = np.load(scene_info["range_depth_path"])
            range_int = np.load(scene_info["range_intensity_path"])
        elif "lidar_path" in scene_info:
            lidar_scan = np.load(scene_info["lidar_path"])
            points = lidar_scan[:, :3].astype(np.float32)
            range_depth, range_int = lidar_converter.points2range(points, labels=lidar_scan[:, 3])
        else:
            raise ValueError("No lidar data found")
        
        # Get range coords of the bbox
        bbox_range_coords = lidar_converter.get_range_coords(bbox_3d)

        # Preprocess range data
        image_depth, image_int, bbox_range_coords, crop_left = lidar_converter.apply_default_transforms(
            range_depth=range_depth,
            range_int=range_int,
            bbox_range_coords=bbox_range_coords,
            image_height=self.image_height,
            image_width=self.image_width,
        )

        # Normalise bbox_range_coords
        bbox_range_coords = bbox_range_coords.astype(np.float32)
        bbox_range_coords[..., 0] /= image_depth.shape[1]
        bbox_range_coords[..., 1] /= image_depth.shape[0]

        # Convert to RGB and resize
        image_depth = np.tile(image_depth[:, :, None], 3)
        image_depth = get_tensor(normalize=False, toTensor=True)(image_depth)
        image_depth = self.resize(image_depth)

        return image_depth, image_int, bbox_range_coords, crop_left, range_depth, range_int