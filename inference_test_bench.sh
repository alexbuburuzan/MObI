python3 scripts/inference_test_bench.py \
    --plms \
    --outdir "results_test_bench_multimodal/dummy" \
    --config "configs/nusc_control_multimodal.yaml" \
    --ckpt "checkpoints/model.ckpt" \
    --scale "5" \
    --ddim_steps "50" \
    --n_samples "1" \
    --compute_metrics \