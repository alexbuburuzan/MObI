cd $WORK_DIR_MOBI

nvidia-smi
conda activate mobi

# Directories
RESULTS_BASE_DIR="./results"
CONFIG_DIR="configs"

# Initialize result table function
initialize_results_table() {
    local run_name="$1"
    local header="$2"
    local results_table="${RESULTS_BASE_DIR}/${run_name}/realism_table.csv"
    mkdir -p "$(dirname "${results_table}")"
    if [ ! -f "${results_table}" ]; then
        echo "${header}" > "${results_table}"
    fi
    echo "${results_table}"
}

# Compute scores function
compute_scores() {
    local out_dir="$1"
    local model_name="$2"
    local ref_type="$3"
    local results_table="$4"

    # FID score
    FID_SCORE=$(python eval_tool/camera/fid_score.py --path_target "${out_dir}/camera/patch_gt" --path_pred "${out_dir}/camera/patch_pred" | grep -oP 'FID:\s*\K[0-9.]+')
    # # LPIPS score
    LPIPS_SCORE=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/camera/patch_gt" --path_pred "${out_dir}/camera/patch_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
    # CLIP score
    CLIP_SCORE=$(python eval_tool/camera/clip_score.py --path_ref "${out_dir}/camera/object_ref" --path_pred "${out_dir}/camera/object_pred" | grep -oP 'CLIP:\s*\K[0-9.]+')

    # LPIPS score for intensity
    echo python eval_tool/camera/lpips_score.py --path_target "${out_dir}/lidar/range_intensity_target" --path_pred "${out_dir}/lidar/range_intensity_pred"
    I_LPIPS=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/lidar/range_intensity_target" --path_pred "${out_dir}/lidar/range_intensity_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
    # LPIPS score for depth
    D_LPIPS=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/lidar/range_depth_target" --path_pred "${out_dir}/lidar/range_depth_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
    echo "${model_name},${ref_type},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE},${D_LPIPS},${I_LPIPS}" >> "${results_table}"
}

# Run experiments with inference
run_experiment() {
    local model_dir="$1"
    local config="$2"
    local run_name="$3"
    local ddim_steps="$4"
    local header="$5"
    local n_samples="$6"

    results_table=$(initialize_results_table "${run_name}" "${header}")
    
    for model_path in ${model_dir}/*.ckpt; do
        model_name=$(basename ${model_path} .ckpt)

        ref_type="track-ref"
        local out_dir="${RESULTS_BASE_DIR}/${run_name}/${model_name}/${ref_type}"

        python3 scripts/inference_test_bench.py \
            --plms \
            --outdir "${out_dir}" \
            --config "${config}" \
            --ckpt "${model_path}" \
            --scale "5" \
            --ddim_steps "${ddim_steps}" \
            --n_samples "${n_samples}" \
            --save_samples \
            --save_visualisations \
            ref_mode="${ref_type}" \
            data.params.test.params.num_samples_per_class=32 \
            use_camera=True \
            use_lidar=True \

        compute_scores "${out_dir}" "${model_name}" "${ref_type}" "${results_table}"
    done
}

run_experiment "models/MObI/512_ablations/2024-12-22T00-13-01_mobi_nusc_512/checkpoints" \
    "${CONFIG_DIR}/mobi_nusc_512.yaml" \
    "512_ablations/2024-12-22T00-13-01_mobi_nusc_512" \
    "50" \
    "Model,Reference Type,FID,LPIPS,CLIP,D-LPIPS,I-LPIPS" \
    "8" \

