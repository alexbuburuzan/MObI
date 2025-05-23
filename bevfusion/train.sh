# RUN_DIR=runs/copy-paste_stop-5_2

# torchpack dist-run -np 8 python \
#     tools/train.py \
#     configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
#     --load_from pretrained/lidar-only-det.pth \
#     --data.samples_per_gpu 2 \
#     --optimizer_config.cumulative_iters 2 \
#     --run-dir ${RUN_DIR}

# torchpack dist-run -np 8 python tools/test.py \
#     configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
#     $RUN_DIR \
#     --eval bbox,map

RUN_DIR=runs/copy-paste_stop-5

# torchpack dist-run -np 8 python \
#     tools/train.py \
#     configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
#     --load_from pretrained/lidar-only-det.pth \
#     --data.samples_per_gpu 2 \
#     --optimizer_config.cumulative_iters 2 \
#     --resume_from "${RUN_DIR}/latest.pth" \
#     # --run-dir ${RUN_DIR}_resume

torchpack dist-run -np 8 python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    ${RUN_DIR}_resume/latest.pth \
    --eval bbox,map
