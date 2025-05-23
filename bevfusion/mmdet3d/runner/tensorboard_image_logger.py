import numpy as np
import torch
import pickle

from mmcv.runner import TensorboardLoggerHook, master_only, HOOKS
from mmdet3d.core.utils import visualize_camera, visualize_lidar


@HOOKS.register_module()
class TensorboardImageLoggerHook(TensorboardLoggerHook):
    @master_only
    def log(self, runner):
        # Log scalars
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

        # Plot scene after all augmentations
        # shape B x num_cameras x C x H x W
        imgs = runner.outputs["info"].get("imgs")

        if imgs is not None:
            imgs = imgs[0]
            
            mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device, dtype=imgs.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device, dtype=imgs.dtype).view(3, 1, 1)

            imgs = imgs * std + mean
            front_images = torch.cat([imgs[2], imgs[0], imgs[1]], axis=2)
            back_images = torch.cat([imgs[4], imgs[3], imgs[5]], axis=2)
            scene = torch.cat([front_images, back_images], axis=1)

            self.writer.add_image(
                "imgs/augmented",
                scene,
                self.get_iter(runner),
                dataformats="CHW"
            )

        # Plot camera figs before copy-paste
        img_figs = runner.outputs["data"].get("img_figs")

        if img_figs is not None:
            img_figs = img_figs[0]
            front_images = torch.cat([img_figs[2], img_figs[0], img_figs[1]], axis=2)
            back_images = torch.cat([img_figs[5], img_figs[3], img_figs[4]], axis=2)
            scene = torch.cat([front_images, back_images], axis=1)

            self.writer.add_image(
                "imgs/scene",
                scene,
                self.get_iter(runner),
                dataformats="CHW"
            )

        # Plot camera figs after copy-paste
        img_figs_cp = runner.outputs["data"].get("img_figs_cp")

        if img_figs_cp is not None:
            img_figs_cp = img_figs_cp[0]
            front_images = torch.cat([img_figs_cp[2], img_figs_cp[0], img_figs_cp[1]], axis=2)
            back_images = torch.cat([img_figs_cp[5], img_figs_cp[3], img_figs_cp[4]], axis=2)
            scene = torch.cat([front_images, back_images], axis=1)

            self.writer.add_image(
                "imgs/copy-paste",
                scene,
                self.get_iter(runner),
                dataformats="CHW"
            )        

        # Plot lidar fig before copy-paste
        lidar_fig = runner.outputs["data"].get("lidar_fig")

        if lidar_fig is not None:
            lidar_fig = lidar_fig[0]
            self.writer.add_image(
                "lidar/scene",
                lidar_fig,
                self.get_iter(runner),
                dataformats="CHW"
            )

        # Plot lidar fig after copy-paste
        lidar_fig_cp = runner.outputs["data"].get("lidar_fig_cp")

        if lidar_fig_cp is not None:
            lidar_figs_cp = lidar_fig_cp[0]
            self.writer.add_image(
                "lidar/copy-paste",
                lidar_figs_cp,
                self.get_iter(runner),
                dataformats="CHW"
            )

            
