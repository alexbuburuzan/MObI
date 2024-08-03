import warnings

import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.stats import beta

import torchvision.transforms as T
import torch.utils.data as data

import torchvision
import albumentations as A

from ldm.data.utils import (
    get_image_coords,
    rotate_bbox,
    translate_bbox,
    get_2d_bbox,
    get_inpaint_mask,
    get_range_inpaint_mask,
    depth_normalization,
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
        ref_mode="id-ref", # id-ref, track-ref, erase-ref, in-domain-ref, cross-domain-ref
        image_height=512,
        image_width=512,
        range_height=512,
        range_width=512,
        reference_image_min_h=100,
        reference_image_max_h=800,
        reference_image_min_w=100,
        reference_image_max_w=1400,
        frustum_iou_max=0.5,
        camera_visibility_min=0.7,
        object_area_crop=0.2,
        object_random_crop=True,
        min_lidar_points=64,
        rot_every_angle=0,
        rot_test_scene=None, # used for rotation test
        use_lidar=False,
        use_camera=True,
        random_range_crop=False,
        num_samples_per_class=None,
        fixed_sampling=True,
        return_original_image=False,
        range_object_norm=False,
        range_object_norm_scale=0.75,
        range_int_norm=False,
        include_erase_boxes=False,
    ) -> None:
        self.state = state
        self.ref_aug = ref_aug
        self.ref_mode = ref_mode
        self.expand_mask_ratio = expand_mask_ratio
        self.expand_ref_ratio = expand_ref_ratio
        self.prob_use_3d_edit_mask = prob_use_3d_edit_mask
        self.rot_test_scene = rot_test_scene
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.random_range_crop = random_range_crop
        self.object_area_crop = object_area_crop
        self.object_random_crop = object_random_crop
        self.return_original_image = return_original_image
        self.range_object_norm = range_object_norm
        self.range_object_norm_scale = range_object_norm_scale
        self.range_int_norm = range_int_norm
        self.num_samples_per_class = num_samples_per_class
        self.fixed_sampling = fixed_sampling

        # Dimensions
        self.image_height = image_height
        self.image_width = image_width
        self.range_height = range_height
        self.range_width = range_width

        self.object_classes = object_classes
        self.objects_meta_all = pd.read_csv(object_database_path, index_col=0)
        # Filter out small, occluded objects
        self.objects_meta_all = self.objects_meta_all[
            ((self.objects_meta_all["reference_image_h"] >= reference_image_min_h) &
             (self.objects_meta_all["reference_image_h"] <= reference_image_max_h) &
             (self.objects_meta_all["reference_image_w"] >= reference_image_min_w) &
             (self.objects_meta_all["reference_image_w"] <= reference_image_max_w) &
             (self.objects_meta_all["max_iou_overlap"] <= frustum_iou_max) &
             (self.objects_meta_all["object_class"].isin(object_classes)) &
             (self.objects_meta_all["camera_visibility_mask"] >= camera_visibility_min) &
             (self.objects_meta_all["num_lidar_points"] >= min_lidar_points))
        ]
        if not include_erase_boxes:
            self.objects_meta_all = self.objects_meta_all[
                (self.objects_meta_all["object_class"] != "empty")
            ]

        # Select an object from each class when testing
        if num_samples_per_class is not None and fixed_sampling:
            self.objects_meta = self.objects_meta_all.groupby("object_class").apply(
                lambda x: x.sample(num_samples_per_class, replace=True)
            )
        else:
            self.objects_meta = self.objects_meta_all
        self.objects_meta = self.objects_meta.reset_index(drop=True)

        self.idx_lists = []
        for object_class in self.object_classes:
            self.idx_lists.append(
                self.objects_meta[self.objects_meta['object_class'] == object_class].index.tolist()
            )

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
                A.Rotate(limit=30, border_mode=0),
                A.Blur(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
            ]
        self.ref_transform = A.Compose(ref_augs)
        self.image_resize = T.Resize([image_height, image_width])

    def __getitem__(self, index):
        if self.num_samples_per_class and self.fixed_sampling is False:
            label = index % len(self.object_classes)
            index = np.random.choice(self.idx_lists[label])
        object_meta = self.objects_meta.iloc[index]

        if self.rot_test_scene is not None:
            scene_info = self.scenes_info[self.rot_test_scene]
            # always use the front camera when rot_test_scene is provided
            cam_idx = 0
        else:
            scene_info = self.scenes_info[object_meta["scene_token"]]
            cam_idx = object_meta["cam_idx"]

        # Reference
        ref_image, ref_bbox_3d, ref_label, ref_class = self.get_reference(object_meta)

        if self.rot_test_scene is None:
            bbox_3d = scene_info["gt_bboxes_3d_corners"][object_meta["scene_obj_idx"]]
        else:
            bbox_3d = translate_bbox(ref_bbox_3d, [0, 9, -1])

        bbox_rot_angle = object_meta.get("bbox_rot_angle", 0)
        bbox_3d = rotate_bbox(bbox_3d, bbox_rot_angle)

        data = {
            "id_name": self.get_id_name(object_meta),
            "bbox_3d": bbox_3d,
            "ref_class": ref_class,
            "image" : {},
            "lidar" : {},
        }

        # Camera
        if self.use_camera:
            data["image"] = self.get_image_data(scene_info, cam_idx, bbox_3d)
            data["image"]["cond"]["ref_image"] = ref_image
            data["image"]["cond"]["ref_label"] = ref_label

        # Lidar
        if self.use_lidar:
            data["lidar"] = self.get_range_data(scene_info, bbox_3d, object_meta["scene_obj_idx"])
            data["lidar"]["cond"]["ref_image"] = ref_image
            data["lidar"]["cond"]["ref_label"] = ref_label

            if self.use_camera:
                data["image"]["cond"]["ref_bbox"][..., 2] = data["lidar"]["cond"]["ref_bbox"][..., 2]

        return data
    
    def __len__(self):
        if self.num_samples_per_class is None or self.fixed_sampling:
            return len(self.objects_meta)
        return len(self.object_classes) * self.num_samples_per_class
    
    def get_reference(self, current_object_meta):
        if self.ref_mode == "id-ref" or self.ref_mode == "erase-ref" or current_object_meta["object_class"] == "empty":
            reference_meta = current_object_meta
        elif self.ref_mode == "in-domain-ref":
            reference_meta = self.objects_meta_all[
                (self.objects_meta_all["object_class"] == current_object_meta["object_class"]) &
                (self.objects_meta_all["is_raining"] == current_object_meta["is_raining"]) &
                (self.objects_meta_all["is_night"] == current_object_meta["is_night"]) &
                (self.objects_meta_all["track_id"] != current_object_meta["track_id"])
            ].sample(1).iloc[0]
        elif self.ref_mode == "cross-domain-ref":
            reference_meta = self.objects_meta_all[
                (self.objects_meta_all["object_class"] == current_object_meta["object_class"]) & (
                (self.objects_meta_all["is_raining"] != current_object_meta["is_raining"]) |
                (self.objects_meta_all["is_night"] != current_object_meta["is_night"]))
            ].sample(1).iloc[0]
        elif self.ref_mode == "track-ref":
            # Sample tracked reference according to a beta distribution given the time difference
            tracked_references = self.objects_meta_all[
                self.objects_meta_all["track_id"] == current_object_meta["track_id"]
            ]
            if len(tracked_references):
                reference_meta = tracked_references.iloc[0]
            else:
                intervals = abs(tracked_references.timestamp - current_object_meta.timestamp)
                intervals /= max(intervals)
                # Sample according to a Beta distribution
                weights = beta.pdf(intervals, 4, 1)
                weights /= np.sum(weights)
                selected_index = np.random.choice(len(intervals), p=weights)

                reference_meta = tracked_references.iloc[selected_index]
        else:
            raise ValueError("Invalid ref_mode")

        ref_obj_idx = reference_meta["scene_obj_idx"]
        cam_idx = reference_meta["cam_idx"]
        ref_scene_info = self.scenes_info[reference_meta["scene_token"]]
        lidar2image = ref_scene_info["lidar2image_transforms"][cam_idx]
        image_path = ref_scene_info["image_paths"][cam_idx]

        ref_bbox_3d = ref_scene_info["gt_bboxes_3d_corners"][ref_obj_idx]
        ref_class = reference_meta["object_class"]
        ref_label = self.object_classes.index(ref_class)

        if self.ref_mode == "erase-ref" or current_object_meta["object_class"] == "empty":
            ref_image = np.zeros((224, 224, 3), dtype=np.uint8)
            ref_class = "empty"
            ref_label = 0
        else:
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

        ref_image = self.ref_transform(image=ref_image)["image"]
        ref_image = Image.fromarray(ref_image)
        ref_image = get_tensor_clip()(ref_image)

        return ref_image, ref_bbox_3d, ref_label, ref_class
    
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
    
    def get_range_data(self, scene_info, bbox_3d, obj_idx):
        lidar_converter = LidarConverter()

        if "range_depth_path" in scene_info and "range_intensity_path" in scene_info:
            range_depth = np.load(scene_info["range_depth_path"])
            range_int = np.load(scene_info["range_intensity_path"])
            range_pitch = np.load(scene_info["range_pitch_path"])
            range_yaw = np.load(scene_info["range_yaw_path"])

            if "range_instance_mask_path" in scene_info:
                range_instance_mask = (np.load(scene_info["range_instance_mask_path"]) == obj_idx).astype(np.float32)
            else:
                range_instance_mask = np.zeros_like(range_depth).astype(np.float32)
                warnings.warn("No instance mask found")
        elif "lidar_path" in scene_info:
            lidar_scan = np.load(scene_info["lidar_path"])
            points = lidar_scan[:, :3].astype(np.float32)
            range_depth, range_int, _, range_pitch, range_yaw = lidar_converter.pcd2range(points, labels=lidar_scan[:, 3])
        else:
            raise ValueError("No lidar data found")
        
        # Get range coords of the bbox
        bbox_range_coords = lidar_converter.get_range_coords(bbox_3d)

        range_depth_orig = range_depth.copy()
        range_int_orig = range_int.copy()
        range_instance_mask_orig = range_instance_mask.copy()

        # Preprocess range data
        range_depth, range_int, range_instance_mask, bbox_range_coords, range_shift_left, width_crop = lidar_converter.apply_default_transforms(
            range_depth=range_depth,
            range_int=range_int,
            mask=range_instance_mask,
            bbox_range_coords=bbox_range_coords,
            height=self.range_height,
            width=self.range_width,
            random_crop=self.random_range_crop,
        )

        # Normalise bbox coords
        bbox_range_coords = torch.tensor(bbox_range_coords).float()
        bbox_range_coords[..., 0] /= self.range_width
        bbox_range_coords[..., 1] /= self.range_height
        min_depth_obj = bbox_range_coords[:, 2].min() - 0.1 * (bbox_range_coords[:, 2].max() - bbox_range_coords[:, 2].min())
        max_depth_obj = bbox_range_coords[:, 2].max() + 0.1 * (bbox_range_coords[:, 2].max() - bbox_range_coords[:, 2].min())
        min_depth_obj = torch.clamp(min_depth_obj, -1, 1)
        max_depth_obj = torch.clamp(max_depth_obj, -1, 1)

        # Normalise range data
        range_depth = get_tensor(normalize=False, toTensor=True)(range_depth)
        if self.range_object_norm:
            # range_depth = torch.tanh((range_depth - center_depth) * self.range_object_norm_scale)
            # bbox_range_coords[..., 2] = torch.tanh((bbox_range_coords[..., 2] - center_depth) * self.range_object_norm_scale)
            range_depth = depth_normalization(range_depth, min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)
            bbox_range_coords[..., 2] = depth_normalization(bbox_range_coords[..., 2], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)

        range_int = ((range_int / 255) - 0.5) * 2
        range_int = get_tensor(normalize=False, toTensor=True)(range_int)
        if self.range_int_norm:
            range_int = 1 - torch.exp(-2 * (range_int + 1))
            range_int = torch.clamp(2 * range_int - 1, -1, 1)
        
        range_data = torch.concat([range_depth, range_int], dim=0)

        # Mask
        range_mask = get_range_inpaint_mask(
            bbox_3d, self.range_height, self.range_width, self.expand_mask_ratio, range_shift_left,
        )
        range_mask = range_mask.unsqueeze(0)
        range_instance_mask = torch.tensor(range_instance_mask).float().unsqueeze(0)

        # Inpainted range
        range_data_inpaint = range_data.clone() * range_mask

        data = {
            "range_data": range_data,
            "range_data_inpaint": range_data_inpaint,
            "range_depth_orig": range_depth_orig,
            "range_int_orig": range_int_orig,
            "range_instance_mask_orig": range_instance_mask_orig,
            "range_shift_left": range_shift_left,
            "width_crop": width_crop,
            "range_mask": range_mask,
            "range_instance_mask": range_instance_mask,
            "range_pitch": range_pitch,
            "range_yaw": range_yaw,
            "min_depth_obj": min_depth_obj,
            "max_depth_obj": max_depth_obj,
            "cond": {
                "ref_bbox": bbox_range_coords,
            }
        }

        return data
    
    def get_image_data(self, scene_info, cam_idx, bbox_3d):
        lidar2image = scene_info["lidar2image_transforms"][cam_idx]
        lidar2camera = scene_info["lidar2camera_transforms"][cam_idx]
        image_path = scene_info["image_paths"][cam_idx]

        # Image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        image = get_tensor()(np.array(image))

        # BBox
        bbox_image_coords = get_image_coords(bbox_3d, lidar2image, include_depth=True)

        # Mask
        use_3d_edit_mask = (random.random() < self.prob_use_3d_edit_mask)
        image_mask = get_inpaint_mask(
            bbox_3d, lidar2image, H, W, self.expand_mask_ratio, use_3d_edit_mask,
        )

        if self.return_original_image:
            image_orig = image.clone()
            image_mask_orig = image_mask.clone()

        if (image_mask == 1).all():
            # Some erase boxes are not projected properly
            image_mask = 1 - image_mask

        mask_coords = torch.nonzero(1 - image_mask)
        y1, x1 = mask_coords.min(dim=0)[0]
        y2, x2 = mask_coords.max(dim=0)[0]

        # Crop and resize
        area = (1 - image_mask).sum().item() / self.object_area_crop
        crop_H = int(np.sqrt(area))
        crop_W = int(np.sqrt(area))

        if y2 - y1 > crop_H:
            crop_W += crop_H - (y2 - y1)
            crop_H = y2 - y1
        if x2 - x1 > crop_W:
            crop_H += crop_W - (x2 - x1)
            crop_W = x2 - x1

        crop_H = min(crop_H, H)
        crop_W = min(crop_W, W)

        if self.object_random_crop:
            left = random.randint(max(0, x2 - crop_W), min(x1, W - crop_W))
            top = random.randint(max(0, y2 - crop_H), min(y1, H - crop_H))
        else:
            left = torch.div(max(0, x2 - crop_W) + min(x1, W - crop_W), 2, rounding_mode='floor')
            top = torch.div(max(0, y2 - crop_H) + min(y1, H - crop_H), 2, rounding_mode='floor')

        image = image[:, top:top+crop_H, left:left+crop_W]
        image_mask = image_mask[top:top+crop_H, left:left+crop_W]

        bbox_image_coords -= np.array([left, top, 0])
        bbox_image_coords[..., 0] /= image.size(2)
        bbox_image_coords[..., 1] /= image.size(1)
        bbox_image_coords = torch.tensor(bbox_image_coords).float()

        image = self.image_resize(image)
        image_mask = self.image_resize(image_mask.unsqueeze(0))


        # Inpainted image
        image_inpaint = image.clone() * image_mask

        data = {
            "GT": image,
            "inpaint_image": image_inpaint,
            "inpaint_mask": image_mask,
            "cond": {
                "ref_bbox": bbox_image_coords,
            }
        }

        if self.return_original_image:
            data["orig"] = {
                "crop": torch.tensor([left, top, crop_W, crop_H]),
                "image": image_orig,
                "mask": image_mask_orig,
                "file_name": image_path.split("/")[-1].split(".")[0],
            }

        return data