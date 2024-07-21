conda activate mobi

python -u main.py \
--logdir models/MObI \
--pretrained_model checkpoints/model.ckpt \
--base configs/nusc_control_multimodal.yaml \
--scale_lr False \
--save_top_k 5 \
--gpus "0,1" \