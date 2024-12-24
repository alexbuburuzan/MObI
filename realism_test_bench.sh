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
    echo ${out_dir}
    # FID score
    FID_SCORE=$(python eval_tool/camera/fid_score.py --path_target "${out_dir}/camera/patch_gt" --path_pred "${out_dir}/camera/patch_pred" | grep -oP 'FID:\s*\K[0-9.]+')
    # LPIPS score
    LPIPS_SCORE=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/camera/patch_gt" --path_pred "${out_dir}/camera/patch_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
    
    # CLIP score
    CLIP_SCORE=$(python eval_tool/camera/clip_score.py --path_ref "${out_dir}/camera/object_ref" --path_pred "${out_dir}/camera/object_pred" | grep -oP 'CLIP:\s*\K[0-9.]+')
    
    if [[ "$5" == "True" ]]; then
        # LPIPS score for intensity
        I_LPIPS=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/lidar/range_intensity_target" --path_pred "${out_dir}/lidar/range_intensity_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
        # LPIPS score for depth
        D_LPIPS=$(python eval_tool/camera/lpips_score.py --path_target "${out_dir}/lidar/range_depth_target" --path_pred "${out_dir}/lidar/range_depth_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+')
        echo "${model_name},${ref_type},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE},${D_LPIPS},${I_LPIPS}" >> "${results_table}"
    else
        echo "${model_name},${ref_type},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE}" >> "${results_table}"
    fi
}

# Run experiments with inference
run_experiment() {
    local model_dir="$1"
    local config="$2"
    local run_name="$3"
    local use_lidar="$4"
    local ddim_steps="$5"
    local header="$6"
    local use_copy_paste="$7"
    local n_samples="$8"

    results_table=$(initialize_results_table "${run_name}" "${header}")
    
    for model_path in ${model_dir}/*.ckpt; do
        model_name=$(basename ${model_path} .ckpt)

        for ref_type in "in-domain-ref" "id-ref" "track-ref" "cross-domain-ref"; do
            local out_dir="${RESULTS_BASE_DIR}/${run_name}/${model_name}/${ref_type}"
            
            # Add --copy-paste flag conditionally
            local copy_paste_flag=""
            if [[ "${use_copy_paste}" == "True" ]]; then
                copy_paste_flag="--copy-paste"
            fi

            python3 scripts/inference_test_bench.py \
                --plms \
                --outdir "${out_dir}" \
                --config "${config}" \
                --ckpt "${model_path}" \
                --scale "5" \
                --ddim_steps "${ddim_steps}" \
                --n_samples "${n_samples}" \
                --save_samples \
                ${copy_paste_flag} \
                --save_visualisations \
                ref_mode="${ref_type}" \
                data.params.test.params.num_samples_per_class=100 \
                use_camera=True \
                use_lidar="${use_lidar}" \

            compute_scores "${out_dir}" "${model_name}" "${ref_type}" "${results_table}" "${use_lidar}"
        done
    done
}

# MObI Experiment
# run_experiment "models/MObI/2024-09-17T21-26-14_nusc_control_multimodal/checkpoints" \
#     "${CONFIG_DIR}/mobi_nusc_512.yaml" \
#     "512_ablations/MObI/2024-09-17T21-26-14_mobi_nusc_512_best" \
#     "True" \
#     "50" \
#     "Model,Reference Type,FID,LPIPS,CLIP,D-LPIPS,I-LPIPS" \
#     "False" \
#     "8"

# # Copy-Paste Experiment (with --copy-paste flag)
# run_experiment "checkpoints" \
#     "${CONFIG_DIR}/pbe.yaml" \
#     "final_results/copy-paste_aux" \
#     "False" \
#     "1" \
#     "Model,Reference Type,FID,LPIPS,CLIP" \
#     "True" \
#     "8" \

# # Paint-by-Example Experiment
# run_experiment "checkpoints" \
#     "${CONFIG_DIR}/pbe.yaml" \
#     "final_results/paint-by-example" \
#     "False" \
#     "50" \
#     "Model,Reference Type,FID,LPIPS,CLIP" \
#     "False" \
#     "8"

