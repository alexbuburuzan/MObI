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
from ldm.data.utils import draw_projected_bbox, visualize_lidar, focus_on_bbox
from ldm.data.box_np_ops import points_in_bbox_corners
from ldm.data.lidar_converter import LidarConverter

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import torchvision
from torchvision.transforms import Resize
from torch.utils.data import ConcatDataset

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
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
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
        "--compute_metrics",
        action="store_true",
        help="compute metrics",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    os.makedirs(camera_path, exist_ok=True)
    os.makedirs(lidar_path, exist_ok=True)

    if opt.rotation_test:
        test_data_config = config.data.params.rotation_test
        test_dataset = instantiate_from_config(test_data_config)
    elif opt.compute_metrics:
        test_data_config = config.data.params.test
        test_dataset = instantiate_from_config(test_data_config)
    else:
        test_data_config = config.data.params.test
        test_data_config['params']['num_sample_per_class'] = 1

        test_data_config['params']['ref_aug'] = False
        test_data_config['params']['ref_mode'] = "same-ref"
        test_dataset = instantiate_from_config(test_data_config)

        test_data_config['params']['ref_aug'] = True
        test_data_config['params']['ref_mode'] = "same-ref"
        test_dataset_aug = instantiate_from_config(test_data_config)
        test_dataset_aug.objects_meta = test_dataset.objects_meta

        test_data_config['params']['ref_aug'] = False
        test_data_config['params']['ref_mode'] = "track-ref"
        test_dataset_track = instantiate_from_config(test_data_config)
        test_dataset_track.objects_meta = test_dataset.objects_meta

        test_data_config['params']['ref_aug'] = False
        test_data_config['params']['ref_mode'] = "random-ref"
        test_dataset_random = instantiate_from_config(test_data_config)
        test_dataset_random.objects_meta = test_dataset.objects_meta

        test_data_config['params']['ref_aug'] = False
        test_data_config['params']['ref_mode'] = "no-ref"
        test_dataset_erase = instantiate_from_config(test_data_config)
        test_dataset_erase.objects_meta = test_dataset.objects_meta

        test_dataset = ConcatDataset([
            test_dataset,
            test_dataset_aug,
            test_dataset_track,
            test_dataset_random,
            test_dataset_erase,
        ])
    

    test_dataloader= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=multiprocessing.cpu_count(), 
        pin_memory=True, 
        shuffle=False,
        #sampler=train_sampler, 
        drop_last=True
    )

    metrics = {}

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
                        inpaint_image=data["z"][:,8:16],
                        inpaint_mask=data["z"][:,[16]]
                    )

                    h_camera, h_lidar = model.decode_sample(samples, data.get("z_lidar"))
                    log, lidar_metrics = model.log_data(batch, data, h_camera, h_lidar, log_metrics=False, split="test")

                    if model.use_camera:
                        if opt.save_samples:
                            pred_grid = log["image_preds"].cpu().numpy()
                            for i in range(batch_size):
                                grid_vis = pred_grid[i].transpose(1, 2, 0)[..., ::-1]
                                cv2.imwrite(os.path.join(camera_path, 'grid-' + segment_id_batch[i] + '_img.png'), grid_vis)

                    if model.use_lidar:
                        if opt.save_samples:
                            pred_grid = log["lidar_input-pred-rec"].cpu().numpy()
                            for i in range(batch_size):
                                grid_vis = pred_grid[i].transpose(1, 2, 0)[..., ::-1]
                                cv2.imwrite(os.path.join(lidar_path, 'grid-' + segment_id_batch[i] + '_lidar.png'), grid_vis)

                        if opt.compute_metrics:
                            for k, v in lidar_metrics.items():
                                if k not in metrics:
                                    metrics[k] = []
                                metrics[k].append(v.item())

    if opt.compute_metrics:
        df = {"mean" : {}, "median" : {}}
        for score_name in metrics:
            metrics[score_name] = np.mean(metrics[score_name])
            if "mean" in score_name:
                df["mean"][score_name.split("/")[-1]] = metrics[score_name]
            elif "median" in score_name:
                df["median"][score_name.split("/")[-1]] = metrics[score_name]

        df = pd.DataFrame(df)
        df.to_csv(os.path.join(outpath, "metrics.csv"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
