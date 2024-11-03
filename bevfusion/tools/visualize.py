import argparse
import copy
import json
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.08)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--edited-samples-path", type=str, default=None)
    parser.add_argument("--edited-objects-list", type=str, default=None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    # Modify cfg
    for entry in cfg.data[args.split]["pipeline"]:
        # We add the token key because we need it for filtering
        if entry["type"] == "Collect3D":
            entry["keys"].append("ann_tokens")
        # Set the number of sweeps to be 1
        if entry["type"] == "LoadPointsFromMultiSweeps":
            entry["sweeps_num"] = 1

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split], dataset_kwargs={"edited_samples_path": args.edited_samples_path})
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    for data in dataflow:
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()
            tokens = data["ann_tokens"].data[0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]
                tokens = tokens[indices]

            # Convert tokens to strings
            tokens = np.array([''.join(chr(i) for i in row) for row in tokens])

            # Filter gt boxes to keep only the ones that were edited
            if args.edited_objects_list is not None:
                with open(os.path.join(dataset.dataset_root, args.edited_objects_list), "r") as f:
                    inserted_boxes = json.load(f)
                if metas["token"] not in inserted_boxes:
                    continue
                indices = [token in inserted_boxes[metas["token"]] for gt_box, token in zip(bboxes, tokens)]
                bboxes = bboxes[indices]
                labels = labels[indices]
                tokens = tokens[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    image,
                    fpath=os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    bboxes=bboxes,
                    labels=labels * 0 + 1 if args.edited_objects_list is not None else labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                    save_figure=True,
                )

        # if "points" in data:
        #     lidar = data["points"].data[0][0].numpy()
        #     visualize_lidar(
        #         lidar,
        #         fpath=os.path.join(args.out_dir, "lidar", f"{name}.png"),
        #         bboxes=bboxes,
        #         # labels=labels,
        #         xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
        #         ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
        #         # classes=cfg.object_classes,
        #     )

        # if masks is not None:
        #     visualize_map(
        #         masks,
        #         fpath=os.path.join(args.out_dir, "map", f"{name}.png"),
        #         classes=cfg.map_classes,
        #     )


if __name__ == "__main__":
    main()
