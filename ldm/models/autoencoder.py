import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config, make_contiguous
from ldm.data.utils import get_lidar_vis, inverse_depth_normalization


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 range_object_norm=False,
                 range_object_norm_scale=0.75,
                 range_int_norm=False,
                 **kwargs):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.range_object_norm = range_object_norm
        self.range_object_norm_scale = range_object_norm_scale
        self.range_int_norm = range_int_norm

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.range_l1_metric = nn.L1Loss(reduction='none')

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        if k == "image":
            x = batch["image"]["GT"]
        elif k == "lidar":
            x = batch["lidar"]["range_data"]
        else:
            raise ValueError("Invalid key")
        
        return make_contiguous(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, split="train", **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            xrec = torch.clamp(xrec, -1., 1.)

            def resize(x, size):
                # return F.interpolate(x, size=size, mode='nearest')
                x = x.permute([0, 2, 3, 1])
                kernel_size = x.shape[1] // size[0]
                x = x.unfold(1, kernel_size, kernel_size)
                x = x.reshape(*x.shape[:3], -1)
                x = x.median(dim=-1).values
                x = x.unsqueeze(1)
                return x

            new_size = (32, xrec.shape[-1])
            
            inpaint_depth = resize(batch["lidar"]["range_data_inpaint"][:, [0, 1]], size=new_size)
            input_depth = resize(batch["lidar"]["range_data"][:, [0, 1]], size=new_size)
            rec_depth = resize(xrec[:, [0, 1]], size=new_size)

            inpaint_int = resize(batch["lidar"]["range_data_inpaint"][:, [2]], size=new_size)
            input_int = resize(batch["lidar"]["range_data"][:, [2]], size=new_size)
            rec_int = resize(xrec[:, [2]], size=new_size)

            mask = resize(batch["lidar"]["range_mask"][:, [0]], size=new_size)
            instance_mask = resize(batch["lidar"]["range_instance_mask"], size=new_size)

            log["range_depth_pred"] = torch.cat([input_depth, inpaint_depth, instance_mask, rec_depth], dim=-2)
            log["range_int_pred"] = torch.cat([input_int, inpaint_int, instance_mask, rec_int], dim=-2)

            if self.range_object_norm:
                # center_depth = batch["lidar"]["center_depth"].view(-1, 1, 1, 1)
                # sample_depth = torch.clamp(torch.atanh(sample_depth) / self.range_object_norm_scale + center_depth, -1, 1)
                # input_depth = torch.clamp(torch.atanh(input_depth) / self.range_object_norm_scale + center_depth, -1, 1)
                # rec_depth = torch.clamp(torch.atanh(rec_depth) / self.range_object_norm_scale + center_depth, -1, 1)

                for i in range(rec_depth.shape[0]):
                    min_depth_obj = batch["lidar"]["min_depth_obj"][i]
                    max_depth_obj = batch["lidar"]["max_depth_obj"][i]
                    input_depth[i] = inverse_depth_normalization(input_depth[i], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)
                    rec_depth[i] = inverse_depth_normalization(rec_depth[i], min_depth_obj, max_depth_obj, alpha=self.range_object_norm_scale)

            if self.range_int_norm:
                input_int = torch.clamp(-0.5 * torch.log(1 - (input_int + 1) / 2) - 1, -1, 1)
                rec_int = torch.clamp(-0.5 * torch.log(1 - (rec_int + 1) / 2) - 1, -1, 1)

            # Compute metrics
            lidar_metrics = {}
            for pred_name,(pred, gt) in {
                "rec_depth": (rec_depth, input_depth),
                "rec_int": (rec_int, input_int),
                }.items():
                for reduction in ["mean", "median"]:
                    B = pred.shape[0]
                    object_scores, mask_scores, full_scores = [], [], []
                    for i in range(B):
                        object_dist = self.range_l1_metric(pred[i][instance_mask[i] == 1], gt[i][instance_mask[i] == 1]).flatten()
                        mask_dist = self.range_l1_metric(pred[i][mask[i] == 0], gt[i][mask[i] == 0]).flatten()
                        full_dist = self.range_l1_metric(pred[i], gt[i]).flatten()

                        if reduction == "mean":
                            object_scores.append(object_dist.mean().item())
                            mask_scores.append(mask_dist.mean().item())
                            full_scores.append(full_dist.mean().item())
                        else:
                            object_scores.append(object_dist.median().item())
                            mask_scores.append(mask_dist.median().item())
                            full_scores.append(full_dist.median().item())

                        if np.isnan(object_scores[-1]):
                            del object_scores[-1]

                    lidar_metrics.update({
                        f"{reduction}/object_{pred_name}_L1" : np.mean(object_scores),
                        f"{reduction}/mask_{pred_name}_L1" : np.mean(mask_scores),
                        f"{reduction}/full_{pred_name}_L1" : np.mean(full_scores),
                    })

            # Make metrics more interpretable by scaling them to the original range
            lidar_metrics = {f"{split}/{k}": v * 25.6 if "depth" in k else v * 128 for k, v in lidar_metrics.items()}
            self.log_dict(lidar_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Point cloud visualisations
            _, input_vis, rec_vis = get_lidar_vis(
                sample=rec_depth,
                input=input_depth,
                rec=rec_depth,
                bboxes=batch["bbox_3d"],
                range_depth_orig=batch["lidar"]["range_depth_orig"],
                range_shift_left=batch["lidar"]["range_shift_left"],
                range_pitch=batch["lidar"]["range_pitch"],
                range_yaw=batch["lidar"]["range_yaw"],
            )

            log["lidar_input-rec"] = torch.cat([input_vis, rec_vis], dim=-2)

            log["mode_samples"] = torch.clamp(self.decode(posterior.mode()), -1., 1.)
            log["input-rec"] = torch.cat([x, xrec], dim=-2)
        else:
            log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
