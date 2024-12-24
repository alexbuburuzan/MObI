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
    # FRD score
    FRD_SCORE=$(python eval_tool/lidar/frd_score.py --path-target "${out_dir}/lidar/range_orig" --path-pred "${out_dir}/lidar/range_pred" | grep -oP 'FRD:\s*\K[0-9.]+')
    echo "${model_name},${ref_type},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE},${FRD_SCORE}" >> "${results_table}"
}

# Run experiments with inference
run_experiment() {
    local model_dir="$1"
    local config="$2"
    local run_name="$3"
    local ddim_steps="$4"
    local header="$5"
    local specific_object="$6"  # Specific object to use
    local rot_every_angle="$7"  # Rotation every angle
    local num_samples_per_class="$8"  # Number of samples per class
    local n_samples="$9"  # Number of samples
    local use_rotation_test="${10}"  # Use rotation test
    local ref_type="${11}"
    local rot_test_scene="${12}"

    results_table=$(initialize_results_table "${run_name}" "${header}")
    
    for model_path in ${model_dir}/*.ckpt; do
        model_name=$(basename ${model_path} .ckpt)

        local out_dir="${RESULTS_BASE_DIR}/${run_name}"

        python3 scripts/inference_test_bench.py \
            --plms \
            --outdir "${out_dir}" \
            --config "${config}" \
            --ckpt "${model_path}" \
            --scale "5" \
            --ddim_steps "${ddim_steps}" \
            --n_samples "${n_samples}" \
            --n_workers "4" \
            --save_samples \
            --save_visualisations \
            $( [ "${use_rotation_test}" == "1" ] && echo "--rotation_test" ) \
            ref_mode="${ref_type}" \
            data.params.rotation_test.params.num_samples_per_class=${num_samples_per_class} \
            $( [ "${use_rotation_test}" == "1" ] && echo "data.params.rotation_test.params.specific_object=\"${specific_object}\"" ) \
            $( [ "${use_rotation_test}" == "1" ] && echo "data.params.rotation_test.params.rot_test_scene=\"${rot_test_scene}\"" ) \
            data.params.rotation_test.params.rot_every_angle=${rot_every_angle} \
            use_camera=True \
            use_lidar=True

        # compute_scores "${out_dir}" "${model_name}" "${ref_type}" "${results_table}"
    done
}

# # Parameters;
# model_dir
# config
# run_name
# ddim_steps
# header
# specific_object  # Specific object to use
# rot_every_angle  # Rotation every angle
# num_samples_per_class  # Number of samples per class
# n_samples  # Number of samples
# use_rotation_test  # Use rotation test
# ref type
# specific scene

# Run the rotation test experiment
run_experiment "models/MObI/2024-09-17T21-26-14_nusc_control_multimodal/checkpoints" \
    "${CONFIG_DIR}/mobi_nusc_512.yaml" \
    "final_visualisations/MObI_512_epoch28/rotation_test_no-inpaint/60_car_red_suv/" \
    "50" \
    "Model,Reference Type,FID,LPIPS,CLIP,FRD" \
    "sample-aaac0015dd724fbfa7c52edc33a64dc1_track-700f31e0225b458ba5205e191a8b8a41_time-1533151258646861_car_id-ref_rot-0_grid_seed42" \
    "60" \
    "64" \
    "1" \
    "1" \
    "id-ref"\
    "a18be1888ea5465eb6530ae4b1eb69d7" \

# # Erase experiment
# run_experiment "models/MObI/2024-09-17T21-26-14_nusc_control_multimodal/checkpoints" \
#     "${CONFIG_DIR}/mobi_nusc_512.yaml" \
#     "final_visualisations/MObI_512_epoch28/erase" \
#     "50" \
#     "Model,Reference Type,FID,LPIPS,CLIP,FRD" \
#     "" \
#     "0" \
#     "32" \
#     "8" \
#     "0" \
#     "erase-ref" \
#     "" \
