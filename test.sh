python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_1.png \
--mask_path examples/mask/example_1.png \
--reference_path examples/reference/example_1.jpg \
--seed 0 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_2.png \
--mask_path examples/mask/example_2.png \
--reference_path examples/reference/example_2.jpg \
--seed 0 \
--scale 5

python scripts/inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_3.png \
--mask_path examples/mask/example_3.png \
--reference_path examples/reference/example_3.jpg \
--seed 0 \
--scale 5

python scripts/inference.py \
--plms --outdir results_nusc \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path /mnt/data/mobi/mit-bevfusion/data/nuscenes/nuscenes_gt_database_pbe/4d8ff5b287bb4e1ba6d0a0f79fc74457_car_14_c27a467cdb1c4643801745a2574d6a46_CAM_FRONT_image.png \
--mask_path /mnt/data/mobi/mit-bevfusion/data/nuscenes/nuscenes_gt_database_pbe/4d8ff5b287bb4e1ba6d0a0f79fc74457_car_14_c27a467cdb1c4643801745a2574d6a46_CAM_FRONT_mask.png \
--reference_path /mnt/data/mobi/mit-bevfusion/data/nuscenes/nuscenes_gt_database_pbe/0915646e387b484785ae2b847ffdee2f_car_47_ec69560d753043d28b2f905dffdea722_CAM_FRONT_reference.png \
--seed 1337 \
--H 256 \
--W 448 \
--scale 5
