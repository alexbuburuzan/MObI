cd $WORK_DIR_MOBI

nvidia-smi
conda activate mobi

python -u main.py \
--logdir models/MObI/512_ablations \
--pretrained_model checkpoints/model.ckpt \
--base configs/mobi_nusc_512.yaml \
--scale_lr False \
--save_top_k 5 \
--gpus "0,1" \