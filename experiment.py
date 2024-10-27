import pickle
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
from tqdm import trange

from pytorch_lightning import seed_everything
from ldm.data.utils import un_norm_clip, un_norm

from ldm.data.nuscenes import NuScenesDataset
from ldm.data.utils import draw_projected_bbox, visualize_lidar, focus_on_bbox
from ldm.data.box_np_ops import points_in_bbox_corners
from ldm.data.lidar_converter import LidarConverter

# Seed everything for reproducibility
seed_everything(41)

data_config = {
    "target": "ldm.data.nuscenes.NuScenesDataset",
    "params": {
        "state": "val",
        "use_lidar": True,
        "use_camera": True,
        "object_database_path": "data/nuscenes/nuscenes_dbinfos_pbe_train.csv",
        "scene_database_path": "data/nuscenes/nuscenes_scene_infos_pbe_train.pkl",
        "expand_mask_ratio": 0.1,
        "expand_ref_ratio": 0,
        "object_area_crop": 0.2,
        "num_samples_per_class": 10,
        "prob_erase_box": 0.3,
        "fixed_sampling": False,
        "ref_aug": True,
        "ref_mode": "track-ref",
        "image_height": 256,
        "image_width": 256,
        "range_height": 256,
        "range_width": 256,
        "object_classes": [
            "car"
        ],
        "random_range_crop": False,
        "range_object_norm": True,
        "range_object_norm_scale": 0.75,
        "range_int_norm": True
    }
}


# Initialize the dataset
dataset = NuScenesDataset(**data_config["params"])

print(len(dataset))

dump_dir = "dump"
os.makedirs(dump_dir, exist_ok=True)

def get_vis(image_tensor, bbox_coords=None, final_size=(512, 512)):
    img = un_norm(image_tensor, size=(512, 512)).cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)[..., ::-1]

    if img.shape[-1] == 1:
        img = (plt.cm.magma(img[..., 0] / 255)[:, :, :3] * 255).astype(np.uint8)[..., ::-1]

    if bbox_coords is not None:
        img = draw_projected_bbox(img, np.array(bbox_coords)[..., :2], thickness=2)

    img = cv2.resize(img, final_size, interpolation=cv2.INTER_LINEAR)
    
    return img

for i in trange(len(dataset)):
    sample = dataset[i]
    bbox_3d = sample["bbox_3d"]

    # show lidar
    depth_vis = get_vis(
        sample["lidar"]["range_data"][[0]],
        sample["lidar"]['cond']['ref_bbox'],
        final_size=(512, 256),
    )
    depth_inpaint_vis = get_vis(
        sample["lidar"]["range_data_inpaint"][[0]],
        # sample["lidar"]['cond']['ref_bbox']
        final_size=(512, 256),
    )
    int_vis = get_vis(
        sample["lidar"]["range_data"][[1]],
        sample["lidar"]['cond']['ref_bbox'],
        final_size=(512, 256),
    )
    int_inpaint_vis = get_vis(
        sample["lidar"]["range_data_inpaint"][[1]],
        # sample["lidar"]['cond']['ref_bbox']
        final_size=(512, 256),
    )

    # show image
    img_vis = get_vis(
        sample["image"]["GT"],
        # sample["image"]['cond']['ref_bbox'],
    )
    img_inpaint_vis = get_vis(
        sample["image"]["inpaint_image"],
        # sample["image"]['cond']['ref_bbox'],
    )
    
    ref_img = sample["lidar"]['cond']['ref_image'].unsqueeze(0)
    ref_img = un_norm_clip(ref_img).squeeze()
    ref_img = (ref_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1]

    # Save lidar vis
    cv2.imwrite(os.path.join(dump_dir, f"range_depth_{i}.png"), depth_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_depth_inpaint_{i}.png"), depth_inpaint_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_int_{i}.png"), int_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_int_inpaint_{i}.png"), int_inpaint_vis)

    # Save image vis
    cv2.imwrite(os.path.join(dump_dir, f"image_{i}.png"), img_vis)
    cv2.imwrite(os.path.join(dump_dir, f"image_inpaint_{i}.png"), img_inpaint_vis)
    cv2.imwrite(os.path.join(dump_dir, f"image_ref_{i}.png"), ref_img)

