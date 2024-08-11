# Run the inference test bench script with the appropriate arguments
python3 scripts/inference_test_bench.py \
    --plms \
    --outdir results/lidar_VAE/adapter \
    --config "configs/nusc_control_multimodal.yaml" \
    --ckpt "models/MObI/2024-08-03T19-20-00_nusc_control_multimodal/checkpoints/epoch=000025.ckpt" \
    --scale "5" \
    --ddim_steps "1" \
    --n_samples "8" \
    --n_workers "4" \
    --save_samples \
    --save_visualisations \
    ref_mode="id-ref" \
    data.params.test.params.num_samples_per_class=100 \
    data.params.test.params.include_erase_boxes=False \
    use_camera=True \
    use_lidar=True \