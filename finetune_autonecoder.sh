conda activate mobi

python -u main.py \
--logdir models/LiDAR \
--pretrained_model checkpoints/image_vae.ckpt \
--base configs/range_autoencoder.yaml \
--scale_lr False \
--gpus "0," \