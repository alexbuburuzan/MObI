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

from omegaconf import OmegaConf

# Seed everything for reproducibility
seed_everything(41)

conf = OmegaConf.load("configs/nusc_control_multimodal.yaml")
data_config = conf["data"]["params"]["train"]

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
    id_name = sample["id_name"]

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
    cv2.imwrite(os.path.join(dump_dir, f"range_depth_{id_name}.png"), depth_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_depth_inpaint_{id_name}.png"), depth_inpaint_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_int_{id_name}.png"), int_vis)
    cv2.imwrite(os.path.join(dump_dir, f"range_int_inpaint_{id_name}.png"), int_inpaint_vis)

    # Save image vis
    cv2.imwrite(os.path.join(dump_dir, f"image_{id_name}.png"), img_vis)
    cv2.imwrite(os.path.join(dump_dir, f"image_inpaint_{id_name}.png"), img_inpaint_vis)
    cv2.imwrite(os.path.join(dump_dir, f"image_ref_{id_name}.png"), ref_img)
