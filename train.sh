python -u main.py \
--logdir models/Paint-by-Example \
--pretrained_model checkpoints/model.ckpt \
--base configs/nusc_control.yaml \
--scale_lr False \
--gpus "0,1,2,3,4,5,6,7" \