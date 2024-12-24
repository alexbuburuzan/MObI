import mmcv
import pandas as pd
import numpy as np
import pickle
from os import path as osp
import multiprocessing
from functools import partial
from tqdm import tqdm, trange
import copy
import random

from mmdet3d.core.utils import visualize_camera
from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.utils import get_frustum, frustum_collision_test, box_collision_test
from ldm.data.lidar_converter import LidarConverter


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def crop_image_patch_no_mask(pos_proposals, org_img):
    num_obj = pos_proposals.shape[0]
    img_patches = []
    for i in range(num_obj):
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        img_patch = org_img[y1:y1 + h, x1:x1 + w]
        img_patches.append(img_patch)
    return img_patches


def create_2d_bbox_mask(pos_proposals, org_img):
    num_obj = pos_proposals.shape[0]
    object_masks = []
    for i in range(num_obj):
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = np.zeros_like(org_img)
        mask_patch[y1:y1 + h, x1:x1 + w] = 1
        object_masks.append(mask_patch)
    return object_masks


def create_3d_bbox_mask(org_img, gt_bboxes_3d, gt_labels_3d, lidar2img):
    num_obj = len(gt_bboxes_3d)
    object_masks = []
    for i in range(num_obj):
        mask = visualize_camera(
            org_img[..., ::-1],
            color=(0, 0, 0),
            bboxes=gt_bboxes_3d[i],
            labels=gt_labels_3d[[i]],
            transform=lidar2img,
            show_image=False,
            filled=True,
        )
        object_masks.append(255 - mask)

    return object_masks


def area(bboxes_2d):
    return (bboxes_2d[:, 2] - bboxes_2d[:, 0]) * (bboxes_2d[:, 3] - bboxes_2d[:, 1])


def process_sample(j, database_save_path):
    global dataset
    input_dict = dataset.get_data_info(j)
    dataset.pre_pipeline(input_dict)
    example = dataset.pipeline(input_dict)

    annos = example["ann_info"]
    sample_idx = example["sample_idx"]
    timestamp = example["timestamp"]
    points = example["points"].tensor.numpy()
    gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
    names = annos["gt_names"]
    name_descriptions = annos["gt_name_descriptions"]
    num_obj = gt_boxes_3d.shape[0]

    city = example["location"].split("-")[0]
    is_raining = ("rain" in example["description"].lower())
    is_night = ("night" in example["description"].lower())

    if num_obj == 0:
        return None, None
    
    lidar_converter = LidarConverter()

    # lidar_path = osp.join(database_save_path, f"sample-{sample_idx}_lidar.npy")
    scene_info = {
        "sample_idx": sample_idx,
        "timestamp": timestamp,
        "location": example["location"],
        "description": example["description"],
        "gt_bboxes_3d": example["ann_info"]["gt_bboxes_3d"].tensor.numpy(),
        "gt_bboxes_3d_corners": example["ann_info"]["gt_bboxes_3d"].corners.numpy(),
        "range_depth_path": osp.join(database_save_path, f"sample-{sample_idx}_range_depth.npy"),
        "range_intensity_path": osp.join(database_save_path, f"sample-{sample_idx}_range_intensity.npy"),
        "range_pitch_path": osp.join(database_save_path, f"sample-{sample_idx}_range_pitch.npy"),
        "range_yaw_path": osp.join(database_save_path, f"sample-{sample_idx}_range_yaw.npy"),
        "range_instance_mask_path": osp.join(database_save_path, f"sample-{sample_idx}_range_instance_mask.npy"),
        "lidar2image_transforms": example["lidar2image"],
        "lidar2camera_transforms": example["lidar2camera"],
        "camera_intrinsics": example["camera_intrinsics"],
        "cam_types": example["cam_types"],
        "image_paths": example["image_paths"],
        "lidar_path": example["lidar_path"],
    }
    range_depth, range_intensity, _, range_pitch, range_yaw = lidar_converter.pcd2range(points[:, :3].astype(np.float32), label=points[:, 3])

    np.save(scene_info["range_depth_path"], range_depth)
    np.save(scene_info["range_intensity_path"], range_intensity)
    np.save(scene_info["range_pitch_path"], range_pitch)
    np.save(scene_info["range_yaw_path"], range_yaw)
    # np.save(lidar_path, points)

    imgs = [np.array(img) for img in example["img"]]
    lidar2image = example["lidar2image"]
    cam_types = example["cam_types"]

    bboxes_3d = annos["gt_bboxes_3d"].corners.numpy()
    bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)

    # Convert each PIL image to ndarray
    imgs = [np.array(_img) for _img in imgs]
    assert len(imgs) == len(lidar2image) and len(imgs) == len(cam_types)

    # Create instance mask for each object
    range_mask = np.zeros(np.prod(range_depth.shape)) - 1
    label = np.arange(0, np.prod(range_depth.shape)).reshape(range_depth.shape)
    points_new, points_label, _ = lidar_converter.range2pcd(range_depth, range_pitch, range_yaw, label)

    object_points = box_np_ops.points_in_bbox_corners(points_new, bboxes_3d[..., :3])
    object_points_orig = box_np_ops.points_in_bbox_corners(points[:, :3], bboxes_3d[..., :3])
    num_lidar_points = []

    for _idx in range(len(bboxes_3d)):
        object_pixels = points_label[object_points[:, _idx]]
        range_mask[object_pixels] = _idx
        num_lidar_points.append(object_points_orig[:, _idx].sum())

    range_mask = range_mask.reshape(range_depth.shape)
    np.save(scene_info["range_instance_mask_path"], range_mask)

    db_object_infos = []

    for _idx, (_img, _lidar2image, cam_type) in enumerate(zip(imgs, lidar2image, cam_types)):
        # visualize_camera(
        #     _img[..., ::-1],
        #     fpath=f"test_img_{_idx}.png",
        #     bboxes=annos["gt_bboxes_3d"],
        #     labels=annos["gt_labels_3d"],
        #     transform=_lidar2image,
        #     classes=dataset.CLASSES,
        #     save_figure=True,
        # )

        # Project 3D bboxes to 2D image space
        coord_img = bboxes_3d @ _lidar2image.T
        coord_img[..., :2] /= coord_img[..., 2, None]
        depth = coord_img[..., 2]
        org_indices = np.arange(coord_img.shape[0])
        visible = (depth > 0).all(axis=-1)
        depth = depth.mean(axis=-1)

        if visible.sum() == 0:
            continue

        coord_img = coord_img[..., :2][visible]
        org_indices = org_indices[visible]
        depth = depth[visible]

        # Extract 2D bboxes using extreme points
        minxy = np.min(coord_img, axis=-2)
        maxxy = np.max(coord_img, axis=-2)
        bboxes_2d = np.concatenate([minxy, maxxy], axis=-1).astype(int)
        bboxes_2d_org = bboxes_2d.copy()
        bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
        bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
        visibility_percentage = area(bboxes_2d) / area(bboxes_2d_org)
        visible = ((bboxes_2d[:, 2:] - bboxes_2d[:, :2]) > 4).all(axis=-1)

        if visible.sum() == 0:
            continue

        bboxes_2d = bboxes_2d[visible]
        org_indices = org_indices[visible]
        depth = depth[visible]
        visibility_percentage = visibility_percentage[visible]

        # frustum IoU-based filtering
        frustums = get_frustum(annos["gt_bboxes_3d"].tensor.numpy())[org_indices]
        frustum_coll_mat = frustum_collision_test(frustums, apply_thresh=False)
        diag = np.arange(frustums.shape[0])
        frustum_coll_mat[diag, diag] = 0
        max_iou_overlap = frustum_coll_mat.max(axis=-1)

        object_img_patches = crop_image_patch_no_mask(bboxes_2d, _img)
        object_3d_masks = create_3d_bbox_mask(
            _img, annos["gt_bboxes_3d"][org_indices], annos["gt_labels_3d"][org_indices], _lidar2image
        )

        for i in range(len(object_img_patches)):
            obj = org_indices[i]
            track_id = annos["ann_tokens"][obj]
            dist = np.sqrt(bboxes_3d[obj,:,0]**2 + bboxes_3d[obj,:,1]**2)

            db_object_infos.append({
                "track_id": track_id,
                "scene_token": sample_idx,
                "timestamp": timestamp,
                "cam_type": cam_type,
                "cam_idx": _idx,
                "scene_obj_idx": obj,
                "object_class": names[obj],
                "name_description": name_descriptions[obj],
                "camera_visibility_2d_box": visibility_percentage[i],
                "num_mask_pixels": (object_3d_masks[i][..., 0] // 255).sum(),
                "max_iou_overlap": max_iou_overlap[i],
                "reference_image_h": object_img_patches[i].shape[0],
                "reference_image_w": object_img_patches[i].shape[1],
                "num_lidar_points": num_lidar_points[obj],
                "city": city,
                "is_raining": is_raining,
                "is_night": is_night,
                "is_erase_box": False,
                "max_distance": dist.max(),
                "min_distance": dist.min()
            })

    return scene_info, db_object_infos


def check_erase_bbox(gt_bboxes_3d):
    gt_frustums = get_frustum(gt_bboxes_3d)
    gt_bboxes_bev = box_np_ops.center_to_corner_box2d(
        gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6]
    )

    # Last box is the erase box
    box_coll_mat = box_collision_test(gt_bboxes_bev, gt_bboxes_bev)
    frustum_coll_mat = frustum_collision_test(gt_frustums[:-1], gt_frustums[[-1]], thresh=0.5)

    coll_mat = np.logical_or(box_coll_mat, frustum_coll_mat)
    diag = np.arange(gt_bboxes_3d.shape[0])
    coll_mat[diag, diag] = False

    return not coll_mat[-1].any()


def create_groundtruth_database(
        dataset_class_name,
        data_path,
        info_prefix,
        info_path=None,
        database_save_path=None,
        db_info_save_path=None,
        scene_info_save_path=None,
        split="train",
        workers=1,
        max_sweeps=0,
        version="v1.0",
    ):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        scene_info_save path (str): Path to save all_scene_infos.
            Default: None.
        split (str): Split of the dataset.
            Default: "train".
        max_sweeps (int): Number of additional LiDAR sweeps.
            Default: 0
    """
    print(f"Create PbE Database of {dataset_class_name} {split} set | workers {workers}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path)

    if dataset_class_name == "NuScenesDataset":
        dataset_cfg.update(
            use_valid_flag=True,
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=True,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=5,
                    use_dim=5),
                dict(
                    type="LoadPointsFromMultiSweeps",
                    sweeps_num=max_sweeps, # no aggregation
                    use_dim=[0, 1, 2, 3, 4],
                    pad_empty_sweeps=True,
                    remove_close=True),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type="LoadMultiViewImageFromFiles",
                    to_float32=True),
            ])
    else:
        raise NotImplementedError

    global dataset
    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_pbe_gt_database_{split}")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_pbe_{split}.csv")
    if scene_info_save_path is None:
        scene_info_save_path = osp.join(data_path, f"{info_prefix}_scene_infos_pbe_{split}.pkl")

    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = []
    all_scene_infos = {}

    with multiprocessing.Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(
                partial(process_sample, database_save_path=database_save_path),
                range(len(dataset))
            ),
            total=len(dataset)
        ))
    # results = [process_sample(0, database_save_path=database_save_path)]

    for scene_info, db_infos in results:
        if scene_info is not None:
            all_scene_infos[scene_info['sample_idx']] = scene_info
            all_db_infos.extend(db_infos)

    # Create data for supervising object removal
    num_scenes = 10000 if split == "train" else 2000
    if "mini" in version:
        num_scenes = num_scenes / 100
    erase_boxes = []
    while len(erase_boxes) < num_scenes:
        object_idx = np.random.randint(0, len(all_db_infos))
        scene_idx = np.random.choice(list(all_scene_infos.keys()))

        object_info = all_db_infos[object_idx]
        source_scene = all_scene_infos[object_info["scene_token"]]
        scene_obj_idx = object_info["scene_obj_idx"]

        # Last box is the erase box
        all_gt_bboxes_3d = np.concatenate([
            all_scene_infos[scene_idx]["gt_bboxes_3d"],
            source_scene["gt_bboxes_3d"][[scene_obj_idx]]
        ])

        all_gt_bboxes_3d_corners = np.concatenate([
            all_scene_infos[scene_idx]["gt_bboxes_3d_corners"],
            source_scene["gt_bboxes_3d_corners"][[scene_obj_idx]]
        ])

        if check_erase_bbox(all_gt_bboxes_3d):
            # Add box to target scene
            all_scene_infos[scene_idx]["gt_bboxes_3d"] = all_gt_bboxes_3d
            all_scene_infos[scene_idx]["gt_bboxes_3d_corners"] = all_gt_bboxes_3d_corners

            erase_box = copy.deepcopy(object_info)

            # Copy selected box to target scene and mark it as empty
            erase_box["scene_token"] = all_scene_infos[scene_idx]["sample_idx"]
            erase_box["is_erase_box"] = True
            erase_box["scene_obj_idx"] = all_scene_infos[scene_idx]["gt_bboxes_3d"].shape[0] - 1
            erase_boxes.append(erase_box)

            if len(erase_boxes) % 200 == 0:
                print("Num erase boxes", len(erase_boxes))

    all_db_infos.extend(erase_boxes)
    all_db_infos_df = pd.DataFrame(all_db_infos)

    # Calculate the visibility percentage of each object
    grouped = all_db_infos_df.groupby(["track_id", "scene_token", "timestamp"])
    total_pixels = grouped["num_mask_pixels"].transform('sum')
    all_db_infos_df['camera_visibility_mask'] = all_db_infos_df['num_mask_pixels'] / total_pixels

    all_db_infos_df.to_csv(db_info_save_path)

    with open(scene_info_save_path, "wb") as f:
        pickle.dump(all_scene_infos, f)

    print(f"{split} PBE database created!")
