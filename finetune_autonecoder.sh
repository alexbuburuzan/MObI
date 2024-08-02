conda activate mobi

python -u main.py \
--logdir models/LiDAR \
--pretrained_model checkpoints/autoencoder/image_vae.ckpt \
--base configs/range_autoencoder.yaml \
--scale_lr False \
--save_top_k 3 \
--gpus "0," \