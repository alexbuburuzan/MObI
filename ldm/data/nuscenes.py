from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Tuple, Union

import os
import cv2
import copy
import pickle
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data

import albumentations as A
import bezier


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


class NuScenesDataset(data.Dataset):
    def __init__(
            self,
            state,
            gt_database_path,
            image_height=512,
            image_width=512,
    ) -> None:
        self.state = state

        with open(gt_database_path, "rb") as f:
            self.objects_meta = pickle.load(f)

        self.ref_augment = A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])
        self.resize=T.Resize([image_height, image_width])

    def __getitem__(self, index):
        object_meta = self.objects_meta[index]
        mask_poly_lines = object_meta["mask_poly_lines"]
        bbox_2d = object_meta["bbox_2d"]

        image = Image.open(object_meta["image_path"]).convert("RGB")
        image_np = np.array(image)
        image_tensor = get_tensor()(image)
        W, H = image.size
   
        ### Get reference image
        bbox_2d_pad = copy.copy(bbox_2d)
        bbox_2d_pad[0] = bbox_2d[0] - min(10, bbox_2d[0])
        bbox_2d_pad[1] = bbox_2d[1] - min(10, bbox_2d[1])
        bbox_2d_pad[2] = bbox_2d[2] + min(10, image.size[0] - bbox_2d[2])
        bbox_2d_pad[3] = bbox_2d[3] + min(10, image.size[1] - bbox_2d[3])

        ref_image_tensor = image_np[
            bbox_2d_pad[1]:bbox_2d_pad[3],
            bbox_2d_pad[0]:bbox_2d_pad[2],
            :
        ]
        ref_image_tensor = self.ref_augment(image=ref_image_tensor)
        ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor = get_tensor_clip()(ref_image_tensor)

        ### Generate mask
        mask_img = np.zeros((H, W))
        cv2.drawContours(mask_img, [mask_poly_lines], 0, 1, -1)
        mask_img = Image.fromarray(mask_img)
        mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_img)
        # TODO: Bezier augmentation, extend mask

        image_tensor_resize = self.resize(image_tensor)
        mask_tensor_resize = self.resize(mask_tensor)
        inpaint_tensor_resize = image_tensor_resize*mask_tensor_resize

        data = {
            "GT": image_tensor_resize,
            "inpaint_image": inpaint_tensor_resize,
            "inpaint_mask": mask_tensor_resize,
            "ref_imgs": ref_image_tensor
        }

        return data

    def __len__(self):
        return len(self.objects_meta)
