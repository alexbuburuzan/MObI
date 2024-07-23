# Run the inference test bench script with the appropriate arguments
python3 scripts/inference_test_bench.py \
    --plms \
    --outdir results/reconstrction/default-1depth-min64-rescaleW \
    --config "configs/nusc_control_multimodal.yaml" \
    --ckpt "checkpoints/model.ckpt" \
    --scale "5" \
    --ddim_steps "1" \
    --n_samples "8" \
    --n_workers "0" \
    --save_samples \
    --save_visualisations \
    ref_mode="same-ref" \
    data.params.test.params.num_samples_per_class=64 \
    use_camera=False \
    use_lidar=True \