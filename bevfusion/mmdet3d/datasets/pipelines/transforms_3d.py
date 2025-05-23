from typing import Any, Dict, Tuple, List
import warnings

import mmcv
import numpy as np
import torch
import torchvision
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
from numpy import random
from PIL import Image
import cv2
import copy

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)
from mmdet.datasets.builder import PIPELINES

from ..builder import OBJECTSAMPLERS
from .utils import noise_per_object_v3_, transform_to_spherical, get_frustum


from mmdet3d.core.utils import visualize_camera, visualize_lidar

@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        return data


@PIPELINES.register_module()
class GlobalRotScaleTrans:
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = random.uniform(*self.resize_lim)
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data


@PIPELINES.register_module()
class GridMask:
    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results["img"]
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results


@PIPELINES.register_module()
class RandomFlip3D:
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = random.choice([0, 1])
        flip_vertical = random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data


@PIPELINES.register_module()
class UnifiedObjectSample(object):
    def __init__(
        self, 
        db_sampler, 
        sample_2d=False,
        sample_method='depth',
        mixup_rate=-1,
        stop_epoch=None,
        visualize_cp=False,
    ) -> None:
        """Sample GT objects to the data.

        Args:
            db_sampler (dict): Config dict of the database sampler.
            sample_2d (bool): Whether to also paste 2D image patch to the images
                This should be true when applying multi-modality cut-and-paste.
                Defaults to True.
            mixup_rate (float)

        """
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.mixup_rate = mixup_rate
        self.epoch = -1
        self.stop_epoch = stop_epoch
        self.visualize_cp = visualize_cp

        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'UnifiedDataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(
            points.coord.numpy(), boxes, origin=(0.5, 0.5, 0.5)
        )
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return input_dict
        
        if self.visualize_cp:
            input_dict['img_figs'], input_dict['lidar_fig'] = self.visualize_scene(input_dict)

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']

        sampled_dict = self.db_sampler.sample_all(
            gt_bboxes_3d.tensor.numpy(),
            gt_labels_3d,
            with_img=self.sample_2d,
        )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_points_num = len(sampled_points)
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points = points.cat([points, sampled_points])

            points_idx = -1 * np.ones(len(points), dtype=np.int)
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict['img']
                lidar2image = input_dict['lidar2image']
                sampled_img = sampled_dict['images']
                sampled_num = len(sampled_gt_bboxes_3d)
                cam_types = input_dict['cam_types']

                imgs, points_keep, bboxes_keep = self.unified_sample(
                    imgs=imgs,
                    cam_types=cam_types,
                    lidar2image=lidar2image,
                    points=points.tensor.numpy(),
                    corners_3d=gt_bboxes_3d.corners.numpy(),
                    bboxes_3d=gt_bboxes_3d,
                    labels_3d=gt_labels_3d,
                    sampled_img=sampled_img,
                    sampled_num=sampled_num,
                    sampled_points_num=sampled_points_num,
                )

        input_dict['img'] = imgs
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d[bboxes_keep]
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)[bboxes_keep]
        input_dict['points'] = points[points_keep]

        if self.visualize_cp:
            input_dict['img_figs_cp'], input_dict['lidar_fig_cp'] = self.visualize_scene(input_dict)

        return input_dict

    def visualize_scene(self, input_dict):
        img_figs = []
        for cam_idx in range(len(input_dict['img'])):
            img = np.array(input_dict['img'][cam_idx])
            img_fig = visualize_camera(
                img[..., ::-1],
                bboxes=input_dict['gt_bboxes_3d'],
                labels=input_dict['gt_labels_3d'],
                transform=input_dict["lidar2image"][cam_idx],
                points=input_dict['points'].tensor.numpy(),
                classes=self.sampler_cfg.classes,
                thickness=2,
            )[..., ::-1]

            # Downsample visualization by 4x
            img_fig = cv2.resize(
                img_fig,
                (img_fig.shape[1] // 2, img_fig.shape[0] // 2),
            )
            img_figs.append(
                img_fig.transpose(2, 0, 1)
            )

        lidar_fig = visualize_lidar(
            lidar=input_dict['points'],
            bboxes=input_dict['gt_bboxes_3d'],
            labels=input_dict['gt_labels_3d'],
            classes=self.sampler_cfg.classes,
            thickness=10,
            radius=10
        )

        # Downsample visualization by 2x
        lidar_fig = cv2.resize(
            lidar_fig, 
            (lidar_fig.shape[1] // 2, lidar_fig.shape[0] // 2),
        )
        lidar_fig = lidar_fig.transpose(2, 0, 1)

        return img_figs, lidar_fig

    def unified_sample(
            self, 
            imgs: List[Image.Image], 
            cam_types: List[str], 
            lidar2image: np.ndarray, 
            points: np.ndarray,
            corners_3d: np.ndarray,
            bboxes_3d: LiDARInstance3DBoxes,
            labels_3d: np.ndarray,
            sampled_img: List[Dict[str, np.ndarray]],
            sampled_num: int,
            sampled_points_num: int,
    ) -> Tuple[List[Image.Image], np.ndarray, np.ndarray]:
        # Bboxes preprocessing
        corners_3d = np.concatenate([corners_3d, np.ones_like(corners_3d[..., :1])], -1)
        orig_coords = bboxes_3d.tensor.numpy()[:, :3]
        is_raw = np.ones(len(corners_3d), dtype=np.bool_)
        is_raw[-sampled_num:] = False
        raw_num = len(is_raw) - sampled_num
        bboxes_keep = np.ones(len(bboxes_3d), dtype=np.bool_)
        frustums = get_frustum(bboxes_3d.tensor.numpy())

        # Point cloud preprocessing
        points_3d = points[:, :4].copy()
        points_3d[:, -1] = 1
        points_keep = np.ones(len(points_3d), dtype=np.bool_)

        # Convert each PIL image to ndarray
        imgs = [np.array(_img) for _img in imgs]
        new_imgs = imgs

        # Paste image patches
        assert len(imgs) == len(lidar2image) and len(sampled_img) == sampled_num and len(cam_types) == len(imgs)
        for _idx, (_img, _lidar2image, cam_type) in enumerate(zip(imgs, lidar2image, cam_types)):
            coord_img = corners_3d @ _lidar2image.T
            coord_img[..., :2] /= coord_img[..., 2, None]
            depth = coord_img[..., 2]

            # filter visible bboxes
            visible = (depth > 0).all(axis=-1)
            img_count = visible.nonzero()[0]

            if visible.sum() == 0:
                continue

            depth = depth.mean(1)[visible]
            coord_img = coord_img[..., :2][visible]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
            visible = ((bbox[:, 2:] - bbox[:, :2]) > 1).all(axis=-1)

            if visible.sum() == 0:
                continue

            depth = depth[visible]
            if 'depth' in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)

            img_count = img_count[visible][paste_order]
            bbox = bbox[visible][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)

            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]:_box[3], _box[0]:_box[2]].copy())

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = raw_img.pop(0)
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + raw_img.pop(
                                0) * self.mixup_rate
                    fg_mask[_box[1]:_box[3], _box[0]:_box[2]] = 1
                else:
                    if cam_type not in sampled_img[_count - raw_num].keys():
                        warnings.warn(
                            "Corresponding image patch of the source object is not available. "
                            "This is likely due to small changes in camera extrinsics between the source and target scenes."
                        )
                        continue

                    img_crop = sampled_img[_count - raw_num][cam_type]
                    if len(img_crop) == 0: continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2, 3]] - _box[[0, 1]]))
                    if self.mixup_rate < 0:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = img_crop
                    else:
                        _img[_box[1]:_box[3], _box[0]:_box[2]] = \
                            _img[_box[1]:_box[3], _box[0]:_box[2]] * (1 - self.mixup_rate) + img_crop * self.mixup_rate

                paste_mask[_box[1]:_box[3], _box[0]:_box[2]] = _count

            new_imgs[_idx] = _img

        # Filter occluded points
        pts_rr = transform_to_spherical(points)

        valid = np.zeros([points.shape[0]], dtype=np.bool_)
        valid_filter = np.zeros([points.shape[0]], dtype=np.bool_)
        valid_sample = np.zeros([points.shape[0]], dtype=np.bool_)
        valid_sample[-sampled_points_num:] = True
        point_indices = box_np_ops.points_in_rbbox(
            points, bboxes_3d.tensor.numpy(), origin=(0.5, 0.5, 0.5)
        )

        depths = np.sqrt(np.square(orig_coords[:, 0]) + np.square(orig_coords[:, 1]) + np.square(orig_coords[:, 2]))
        idxs = np.argsort(depths)

        for idx in idxs:
            cur_frus = frustums[idx]

            # valid points in object frustum
            val = (pts_rr[:, 1] > cur_frus[1, 0, 0]) & (pts_rr[:, 1] < cur_frus[1, 1, 0])
            sp_frus = [cur_frus[2, :, 0]] if cur_frus[2, 0, 1] < 0 else [cur_frus[2, :, 0], cur_frus[2, :, 1]]
            for frus in sp_frus:
                val = val & (pts_rr[:, 2] > frus[0]) & (pts_rr[:, 2] < frus[1])

            val1 = (point_indices[:, idx]) & (valid_filter < 1)  # points in 3D box and not filtered
            valid[val1] = 1  # remained points of current object - valid set to 1
            val = val & (np.logical_not(valid))
            if is_raw[idx]:  # sampled box -> filter bg and fg; original box -> only filter sampled fg
                val = val & valid_sample

            valid_filter[val] = 1

        points_keep = valid_filter < 1

        # Filter bboxes with no points
        for i in range(len(bboxes_keep)):
            val = (valid_filter < 1) & (point_indices[:, i])
            if not val.any():
                bboxes_keep[i] = False

        # Convert each ndarray back to PIL Image
        new_imgs = [Image.fromarray(_img) for _img in new_imgs]
        return new_imgs, points_keep, bboxes_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectPaste:
    """Sample GT objects to the data.
    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, stop_epoch=None):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.epoch = -1
        self.stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.
        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.
        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(
            points.coord.numpy(), boxes, origin=(0.5, 0.5, 0.5)
        )
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, data):
        """Call function to sample ground truth objects to the data.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        # change to float for blending operation
        points = data["points"]
        if self.sample_2d:
            # NOTE: Not supported
            img = data["img"]
            gt_bboxes_2d = data["gt_bboxes"]
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img,
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]
                ).astype(np.float32)

                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        data["points"] = points

        return data


@PIPELINES.register_module()
class ObjectNoise:
    """Apply noise to each GT objects in the scene.
    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(
        self,
        translation_std=[0.25, 0.25, 0.25],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.15707963267, 0.15707963267],
        num_try=100,
    ):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, data):
        """Call function to apply noise to each ground truth in the scene.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = data["gt_bboxes_3d"]
        points = data["points"]

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try,
        )

        data["gt_bboxes_3d"] = gt_bboxes_3d.new_box(numpy_box)
        data["points"] = points.new_point(numpy_points)
        return data


@PIPELINES.register_module()
class FrameDropout:
    def __init__(self, prob: float = 0.5, time_dim: int = -1) -> None:
        self.prob = prob
        self.time_dim = time_dim

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        offsets = []
        for offset in torch.unique(data["points"].tensor[:, self.time_dim]):
            if offset == 0 or random.random() > self.prob:
                offsets.append(offset)
        offsets = torch.tensor(offsets)

        points = data["points"].tensor
        indices = torch.isin(points[:, self.time_dim], offsets)
        data["points"].tensor = points[indices]
        return data


@PIPELINES.register_module()
class PointShuffle:
    def __call__(self, data):
        data["points"].shuffle()
        return data


@PIPELINES.register_module()
class ObjectRangeFilter:
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter objects by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            data["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(data["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data


@PIPELINES.register_module()
class ObjectNameFilter:
    """Filter GT objects by their names.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, data):
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        return data


@PIPELINES.register_module()
class PointSample:
    """Point sample.
    Sampling data to a certain number.
    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(
        self,
        points,
        num_samples,
        sample_range=None,
        replace=False,
        return_choices=False,
    ):
        """Points random sampling.
        Sample points to a certain number.
        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = points.shape[0] < num_samples
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            depth = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(depth > sample_range)[0]
            near_inds = np.where(depth <= sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, data):
        """Call function to sample points to in indoor scenes.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        # Points in Camera coord can provide the depth information.
        # TODO: Need to suport distance-based sampling for other coord system.
        if self.sample_range is not None:
            from mmdet3d.core.points import CameraPoints

            assert isinstance(
                points, CameraPoints
            ), "Sampling based on distance is only appliable for CAMERA coord"
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True,
        )
        data["points"] = points
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(num_points={self.num_points},"
        repr_str += f" sample_range={self.sample_range},"
        repr_str += f" replace={self.replace})"

        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter:
    """Filter background points near the bounding box.
    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (
            is_tuple_of(bbox_enlarge_range, float) and len(bbox_enlarge_range) == 3
        ) or isinstance(
            bbox_enlarge_range, float
        ), f"Invalid arguments bbox_enlarge_range {bbox_enlarge_range}"

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(bbox_enlarge_range, dtype=np.float32)[
            np.newaxis, :
        ]

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        gt_bboxes_3d = data["gt_bboxes_3d"]

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5)
        )
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5)
        )
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks, enlarge_foreground_masks)

        data["points"] = points[valid_masks]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler:
    """Voxel based point sampler.
    Apply voxel sampling to multiple sweep points.
    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg["max_num_points"] == cur_sweep_cfg["max_num_points"]
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.
        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points
        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros(
                [
                    sampler._max_voxels - voxels.shape[0],
                    sampler._max_num_points,
                    point_dim,
                ],
                dtype=points.dtype,
            )
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results["points"]
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results["pts_mask_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results["pts_mask_fields"])
        for idx, key in enumerate(results["pts_seg_fields"]):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = points_numpy[:, self.time_dim] == 0
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(
            cur_sweep_points, self.cur_voxel_generator, points_numpy.shape[1]
        )
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(
                prev_sweeps_points, self.prev_voxel_generator, points_numpy.shape[1]
            )

            points_numpy = np.concatenate([cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results["points"] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split("\n")
            repr_str = [" " * indent + t + "\n" for t in repr_str]
            repr_str = "".join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += "(\n"
        repr_str += " " * indent + f"num_cur_sweep={self.cur_voxel_num},\n"
        repr_str += " " * indent + f"num_prev_sweep={self.prev_voxel_num},\n"
        repr_str += " " * indent + f"time_dim={self.time_dim},\n"
        repr_str += " " * indent + "cur_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.cur_voxel_generator), 8)},\n"
        repr_str += " " * indent + "prev_voxel_generator=\n"
        repr_str += f"{_auto_indent(repr(self.prev_voxel_generator), 8)})"
        return repr_str


@PIPELINES.register_module()
class ImagePad:
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


@PIPELINES.register_module()
class ImageDistort:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        data["img"] = new_imgs
        return data
