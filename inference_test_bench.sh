conda activate mobi

MODEL_DIR="models/MObI/2024-08-03T19-20-00_nusc_control_multimodal/checkpoints"
RUN_NAME="2024-08-03T19-20-00_nusc_control_multimodal_25"

# Initialize the results table
RESULTS_TABLE="./results/${RUN_NAME}/realism_table.csv"
mkdir -p "$(dirname "${RESULTS_TABLE}")"
if [ ! -f "${RESULTS_TABLE}" ]; then
    # If the file does not exist, create it and add the header
    echo "Model,Reference Type,FID,LPIPS,CLIP" > "${RESULTS_TABLE}"
fi

# Loop over each model file in the MODEL_DIR that ends with .ckpt
for MODEL_PATH in ${MODEL_DIR}/*.ckpt; do
    # Extract the model name from the file path
    echo $MODEL_PATH
    MODEL_NAME=$(basename ${MODEL_PATH} .ckpt)
    
    # Loop over the reference types
    for REF_TYPE in "id-ref" "track-ref" "in-domain-ref" "cross-domain-ref"; do
        # Set the output directory based on the model name and reference type
        OUT_DIR="./results/${RUN_NAME}/${MODEL_NAME}/${REF_TYPE}"

        # Run the inference test bench script with the appropriate arguments
        python3 scripts/inference_test_bench.py \
            --plms \
            --outdir "${OUT_DIR}" \
            --config "configs/nusc_control_multimodal.yaml" \
            --ckpt "${MODEL_PATH}" \
            --scale "5" \
            --ddim_steps "50" \
            --n_samples "8" \
            --save_samples \
            --save_visualisations \
            ref_mode="${REF_TYPE}" \
            data.params.test.params.num_samples_per_class=100 \
            use_camera=True \
            use_lidar=True \

        # FID score
        FID_OUTPUT=$(python eval_tool/camera/fid_score.py --path_target "${OUT_DIR}/camera/patch_gt" --path_pred "${OUT_DIR}/camera/patch_pred")
        FID_SCORE=$(echo "${FID_OUTPUT}" | grep -oP 'FID:\s*\K[0-9.]+')

        # LPIPS score
        LPIPS_OUTPUT=$(python eval_tool/camera/lpips_score.py --path_target "${OUT_DIR}/camera/patch_gt" --path_pred "${OUT_DIR}/camera/patch_pred")
        LPIPS_SCORE=$(echo "${LPIPS_OUTPUT}" | grep -oP 'LPIPS:\s*\K[0-9.]+')

        # CLIP score
        CLIP_OUTPUT=$(python eval_tool/camera/clip_score.py --path_ref "${OUT_DIR}/camera/object_ref" --path_pred "${OUT_DIR}/camera/object_pred")
        CLIP_SCORE=$(echo "${CLIP_OUTPUT}" | grep -oP 'CLIP:\s*\K[0-9.]+')

        # Append the results to the table
        echo "${MODEL_NAME},${REF_TYPE},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE}" >> ${RESULTS_TABLE}
    done
done

# # Display the results table
# cat "${RESULTS_TABLE}"


# #!/bin/bash --login
# #$ -cwd
# #$ -l a100=1
# #$ -pe smp.pe 12

# conda activate mobi

# # MODEL_DIR="models/MObI/2024-08-06T10-29-59_nusc_control_multimodal/checkpoints"
# # RUN_NAME="2024-08-06T10-29-59_nusc_control_multimodal_best"

# MODEL_DIR="checkpoints"
# RUN_NAME="Copy-and-paste"

# # Initialize the results table
# RESULTS_TABLE="./results/${RUN_NAME}/realism_table.csv"
# mkdir -p "$(dirname "${RESULTS_TABLE}")"
# if [ ! -f "${RESULTS_TABLE}" ]; then
#     # If the file does not exist, create it and add the header
#     echo "Model,Reference Type,FID,LPIPS,CLIP" > "${RESULTS_TABLE}"
# fi

# # Loop over each model file in the MODEL_DIR that ends with .ckpt
# for MODEL_PATH in ${MODEL_DIR}/*.ckpt; do
#     # Extract the model name from the file path
#     echo $MODEL_PATH
#     MODEL_NAME=$(basename ${MODEL_PATH} .ckpt)
    
#     # Loop over the reference types
#     for REF_TYPE in "id-ref" "track-ref" "in-domain-ref" "cross-domain-ref"; do
#         # Set the output directory based on the model name and reference type
#         OUT_DIR="./results/${RUN_NAME}/${MODEL_NAME}/${REF_TYPE}"

#         # Run the inference test bench script with the appropriate arguments
#         python3 scripts/inference_test_bench.py \
#             --plms \
#             --outdir "${OUT_DIR}" \
#             --config "configs/pbe.yaml" \
#             --ckpt "${MODEL_PATH}" \
#             --scale "5" \
#             --ddim_steps "50" \
#             --n_samples "8" \
#             --save_samples \
#             --save_visualisations \
#             --copy-paste \
#             ref_mode="${REF_TYPE}" \
#             data.params.test.params.num_samples_per_class=100 \
#             data.params.test.params.expand_mask_ratio=0 \
#             use_camera=True \
#             use_lidar=False \

#         # FID score
#         FID_OUTPUT=$(python eval_tool/camera/fid_score.py --path_target "${OUT_DIR}/camera/patch_gt" --path_pred "${OUT_DIR}/camera/patch_pred")
#         FID_SCORE=$(echo "${FID_OUTPUT}" | grep -oP 'FID:\s*\K[0-9.]+')

#         # LPIPS score
#         LPIPS_OUTPUT=$(python eval_tool/camera/lpips_score.py --path_target "${OUT_DIR}/camera/patch_gt" --path_pred "${OUT_DIR}/camera/patch_pred")
#         LPIPS_SCORE=$(echo "${LPIPS_OUTPUT}" | grep -oP 'LPIPS:\s*\K[0-9.]+')

#         # CLIP score
#         CLIP_OUTPUT=$(python eval_tool/camera/clip_score.py --path_ref "${OUT_DIR}/camera/object_ref" --path_pred "${OUT_DIR}/camera/object_pred")
#         CLIP_SCORE=$(echo "${CLIP_OUTPUT}" | grep -oP 'CLIP:\s*\K[0-9.]+')

#         # Append the results to the table
#         echo "${MODEL_NAME},${REF_TYPE},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE}" >> ${RESULTS_TABLE}
#     done
# done

# # Display the results table
# cat "${RESULTS_TABLE}"
