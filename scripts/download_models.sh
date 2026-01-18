cd $WORK_DIR_MOBI

mkdir -p checkpoints && cd checkpoints
# Paint-by-Example
wget https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt

# MObI
mkdir -p mobi_nusc_512 && cd mobi_nusc_512
wget https://huggingface.co/alexbuburuzan/MObI/resolve/main/mobi_nuscenes_epoch28.ckpt

mkdir -p autoencoders && cd autoencoders
# Range autoencoder
wget https://huggingface.co/alexbuburuzan/MObI/resolve/main/range_autoencoder.ckpt
