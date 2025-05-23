torchpack dist-run -np 8 python tools/test.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    runs/experiment2/latest.pth \
    --eval bbox,map \
