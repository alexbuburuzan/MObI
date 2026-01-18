cd $WORK_DIR_MOBI

nvidia-smi
conda activate mobi

# Directories and configurations
MODEL_DIR="models/MObI/2024-09-17T21-26-14_nusc_control_multimodal/checkpoints"
CONFIG_PATH="configs/mobi_nusc_512.yaml"
RUN_NAME="final_results/MObI_512_epoch28"
N_SAMPLES=8

# Loop through all model checkpoints and reference types
for MODEL_PATH in ${MODEL_DIR}/*.ckpt; do
    MODEL_NAME=$(basename ${MODEL_PATH} .ckpt)

    REF_TYPE="track-ref"
    OUT_DIR="./results/${RUN_NAME}/${MODEL_NAME}/${REF_TYPE}_detection_all"
    
    # Run inference for the specified configuration
    python3 scripts/inference_test_bench.py \
        --plms \
        --outdir "${OUT_DIR}" \
        --config "${CONFIG_PATH}" \
        --ckpt "${MODEL_PATH}" \
        --scale "5" \
        --ddim_steps "50" \
        --n_samples "${N_SAMPLES}" \
        --save_samples \
        --save_visualisations \
        data.params.test.params.object_meta_dump_path="${OUT_DIR}/objects.json" \
        data.params.test.params.camera_visibility_min=1 \
        data.params.test.params.num_samples_per_class=200 \
        ref_mode="${REF_TYPE}" \
        use_camera=True \
        use_lidar=True
done