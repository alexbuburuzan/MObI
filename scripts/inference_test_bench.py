import argparse, os
import cv2
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
# from imwatermark import WatermarkEncoder
from itertools import islice
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import point_cloud_utils as pcu
import multiprocessing

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.utils import postprocess_range_depth_int, un_norm_clip
from ldm.data.lidar_converter import LidarConverter
from ldm.data.box_np_ops import points_in_bbox_corners

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import torchvision
from torchvision.transforms import Resize
from torch.utils.data import ConcatDataset
import torch.nn.functional as F

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def move_to_device(batch, device):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
            elif isinstance(batch[k], dict):
                batch[k] = move_to_device(batch[k], device)
        return batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="number of workers",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--rotation_test",
        action="store_true",
        help="insert object for rotated bbox",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
    )
    parser.add_argument(
        "--save_visualisations",
        action="store_true",
    )
    parser.add_argument(
        "--copy-paste",
        action="store_true",
    )
    parser.add_argument(
        'overrides',
        nargs=argparse.REMAINDER,
        help='Configuration overrides',
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(f"{opt.config}")
    cli_conf = OmegaConf.from_dotlist(opt.overrides)
    config = OmegaConf.merge(config, cli_conf)
    
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    lidar_path = os.path.join(outpath, "lidar")
    camera_path = os.path.join(outpath, "camera")
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(camera_path, exist_ok=True)
    os.makedirs(lidar_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    if opt.rotation_test:
        test_data_config = config.data.params.rotation_test
        test_data_config["params"]["return_original_image"] = opt.save_samples
        test_dataset = instantiate_from_config(test_data_config) 
    else:
        test_data_config = config.data.params.test
        test_data_config["params"]["return_original_image"] = opt.save_samples
        test_dataset = instantiate_from_config(test_data_config)
    

    test_dataloader= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=opt.n_workers, 
        pin_memory=True, 
        shuffle=False,
        #sampler=train_sampler, 
        drop_last=False
    )

    metrics = {}

    if opt.copy_paste:
        opt.ddim_steps = 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, model.channels, model.image_size, model.image_size], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for batch in tqdm(test_dataloader):
                    if opt.rotation_test:
                        # diffusion models are highly sensitive to the initial noise
                        # this should help with consistency
                        seed_everything(opt.seed)

                    segment_id_batch = batch["id_name"]
                    batch = move_to_device(batch, device)

                    data = model.get_input(
                        batch,
                        model.first_stage_key,
                        force_c_encode=True,
                        return_vae_rec=True,
                    )

                    uc = None
                    if opt.scale != 1.0:
                        uc = [model.learnable_vector.repeat(data['z'].shape[0],1,1)]
                        if "ref_bbox" in model.cond_stage_key:
                            uc.append(model.bbox_uncond_vector.repeat(data['z'].shape[0],1,1))
                        uc = torch.cat(uc, dim=1)
                    c = data["cond"]

                    shape = [model.channels, model.image_size, model.image_size]
                    if opt.plms:
                        samples, _ = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=data["z"].shape[0],
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                            inpaint_image=data["z"][:,4:8],
                            inpaint_mask=data["z"][:,[8]]
                        )
                    else:
                        samples, _ = sampler.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            batch_size=data["z"].shape[0],
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                            test_model_kwargs={
                                "inpaint_image": data["z"][:,4:8],
                                "inpaint_mask": data["z"][:,[8]]
                            }
                        )

                    h_camera, h_lidar = model.decode_sample(samples, data.get("z_lidar"))
                    log, lidar_metrics = model.log_data(batch, data, h_camera, h_lidar, log_metrics=False, return_sample=opt.save_samples, split="test")
                    num_samples = len(batch["bbox_3d"])

                    if model.use_camera:
                        pred_grid = log["image_preds"].cpu().numpy()
                        pred_grid_no_box = log["image_preds_no_box"].cpu().numpy()
                        for i in range(num_samples):
                            if opt.save_visualisations:
                                grid_vis = pred_grid[i].transpose(1, 2, 0)[..., ::-1]
                                grid_vis_no_box = pred_grid_no_box[i].transpose(1, 2, 0)[..., ::-1]
                                vis = np.concatenate([grid_vis, grid_vis_no_box], axis=1)
                                os.makedirs(os.path.join(camera_path, "grid"), exist_ok=True)
                                cv2.imwrite(os.path.join(camera_path, "grid", segment_id_batch[i] + f'_grid_seed{opt.seed}.png'), vis)

                            if opt.save_samples:
                                patch_pred = log["image_sample"][[i]]
                                patch_gt = batch["image"]["GT"][[i]]
                                object_ref = batch["image"]["cond"]["ref_image"][[i]]
                                image = batch["image"]["orig"]["image"][i].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                mask = batch["image"]["orig"]["mask"][i].cpu().numpy()
                                file_name = batch["image"]["orig"]["file_name"][i]
                                left, top, crop_W, crop_H = batch["image"]["orig"]["crop"][i]

                                mask_coords = np.nonzero(1 - mask)
                                y1, y2 = mask_coords[0].min(), mask_coords[0].max()
                                x1, x2 = mask_coords[1].min(), mask_coords[1].max()

                                object_ref = un_norm_clip(object_ref, size=(224, 224))
                                object_ref = object_ref[0].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                object_ref = (object_ref * 255).astype(np.uint8)

                                patch_gt = F.interpolate(patch_gt, (crop_H, crop_W), mode='bilinear')
                                patch_gt = patch_gt[0].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                patch_gt = (((patch_gt + 1.0) / 2.0) * 255).astype(np.uint8)

                                patch_pred = F.interpolate(patch_pred, (crop_H, crop_W), mode='bilinear')
                                patch_pred = patch_pred[0].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                patch_pred = (((patch_pred + 1.0) / 2.0) * 255).astype(np.uint8)

                                image_pred = np.zeros_like(image)
                                image_pred[top:top+crop_H, left:left+crop_W] = patch_pred
                                if opt.copy_paste:
                                    image_pred[y1:y2, x1:x2, :] = cv2.resize(object_ref, (x2 - x1, y2 - y1))
                                    mask_convolved = mask
                                else:
                                    mask_convolved = cv2.GaussianBlur(mask, (15, 15), 7.0)

                                image = (((image + 1.0) / 2.0) * 255).astype(np.uint8)
                                image_recon = mask_convolved[..., None] * image + (1 - mask_convolved[..., None]) * image_pred

                                composited_patch_pred = image_recon[top:top+crop_H, left:left+crop_W]

                                object_pred = cv2.resize(image_pred[y1:y2, x1:x2, :], (224, 224))

                                # save samples
                                cv2.imwrite(os.path.join(sample_path, file_name), image_recon)

                                for file in "object_pred", "object_ref", "patch_gt", "patch_pred":
                                    os.makedirs(os.path.join(camera_path, file), exist_ok=True)

                                cv2.imwrite(os.path.join(camera_path, "object_pred", f"{segment_id_batch[i]}_object_pred_seed{opt.seed}.png"), object_pred)
                                cv2.imwrite(os.path.join(camera_path, "object_ref", f"{segment_id_batch[i]}_object_ref_seed{opt.seed}.png"), object_ref)
                                cv2.imwrite(os.path.join(camera_path, "patch_gt", f"{segment_id_batch[i]}_gt_seed{opt.seed}.png"), patch_gt)
                                cv2.imwrite(os.path.join(camera_path, "patch_pred", f"{segment_id_batch[i]}_pred_seed{opt.seed}.png"), composited_patch_pred)

                    if model.use_lidar:
                        for i in range(num_samples):
                            if opt.save_visualisations:
                                pcd_vis = log["lidar_input-pred-rec"][i].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                os.makedirs(os.path.join(lidar_path, "point_clouds"), exist_ok=True)
                                cv2.imwrite(os.path.join(lidar_path, "point_clouds", segment_id_batch[i] + f'_grid_pc_seed{opt.seed}.png'), pcd_vis)

                                range_depth_vis = log["range_depth_pred"][i].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                range_depth_vis = ((range_depth_vis + 1.0) / 2.0 * 255).astype(np.uint8)
                                os.makedirs(os.path.join(lidar_path, "range_depth"), exist_ok=True)
                                cv2.imwrite(os.path.join(lidar_path, "range_depth", segment_id_batch[i] + f'_grid_depth_seed{opt.seed}.png'), range_depth_vis)

                                range_int_vis = log["range_int_pred"][i].cpu().numpy().transpose(1, 2, 0)[..., ::-1]
                                range_int_vis = ((range_int_vis + 1.0) / 2.0 * 255).astype(np.uint8)
                                os.makedirs(os.path.join(lidar_path, "range_intensity"), exist_ok=True)
                                cv2.imwrite(os.path.join(lidar_path, "range_intensity", segment_id_batch[i] + f'_grid_intensity_seed{opt.seed}.png'), range_int_vis)

                        if opt.save_samples:
                            pitch = batch['lidar']["range_pitch"].cpu().numpy()
                            yaw = batch['lidar']["range_yaw"].cpu().numpy()

                            range_sample_depth, range_sample_int = postprocess_range_depth_int(
                                range_depth=log["range_sample_depth"],
                                range_depth_orig=batch["lidar"]["range_depth_orig"],
                                range_int=log["range_sample_int"],
                                range_int_orig=batch["lidar"]["range_int_orig"],
                                crop_left=batch["lidar"]["range_shift_left"],
                                width_crop=batch["lidar"]["width_crop"],
                            )

                            lidar_converter = LidarConverter()
                            for i in range(num_samples):
                                bbox_3d = batch["bbox_3d"][[i]].cpu().numpy()
                                gt_instance_mask = batch["lidar"]["range_instance_mask_orig"][i].cpu().numpy()
                                file_name = batch["lidar"]["file_name"][i]

                                # create instance mask for predicted object
                                pred_instance_mask = np.zeros(np.prod(gt_instance_mask.shape))
                                label = np.arange(0, np.prod(gt_instance_mask.shape)).reshape(gt_instance_mask.shape)
                                points, points_label = lidar_converter.range2pcd(range_sample_depth[i], pitch[i], yaw[i], label)

                                object_points = points_in_bbox_corners(points, bbox_3d)
                                object_pixels = points_label[object_points[:, 0]]
                                pred_instance_mask[object_pixels] = 1
                                pred_instance_mask = pred_instance_mask.reshape(gt_instance_mask.shape)

                                # paste object
                                instance_mask = np.logical_or(pred_instance_mask, gt_instance_mask)

                                range_depth_final = np.where(
                                    instance_mask,
                                    range_sample_depth[i],
                                    batch["lidar"]["range_depth_orig"][i].cpu().numpy()
                                )

                                range_int_final = np.where(
                                    instance_mask,
                                    range_sample_int[i],
                                    batch["lidar"]["range_int_orig"][i].cpu().numpy()
                                )

                                # create edited point cloud
                                points_coord, points_int = lidar_converter.range2pcd(range_depth_final, pitch[i], yaw[i], range_int_final)
                                pred_points = np.concatenate([points_coord, points_int[:, None]], axis=1)

                                np.save(os.path.join(sample_path, file_name), pred_points)

                        for k, v in lidar_metrics.items():
                            if k not in metrics:
                                metrics[k] = []
                            if not np.isnan(v):
                                metrics[k].append(v.item())

    df = {"mse" : {}, "median_error" : {}}
    for score_name in metrics:
        metrics[score_name] = np.mean(metrics[score_name])
        if "mse" in score_name:
            df["mse"][score_name.split("/")[-1]] = metrics[score_name]
        elif "median_error" in score_name:
            df["median_error"][score_name.split("/")[-1]] = metrics[score_name]

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(outpath, "metrics.csv"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
