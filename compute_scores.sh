ROOT="./results_test_bench/pbe_camera-only_512"

python eval_tool/fid/fid_score.py "${ROOT}/camera/patch_gt" "${ROOT}/camera/patch_pred"
python eval_tool/clip_score/region_clip_score.py --path_ref "${ROOT}/camera/object_ref" --path_pred "${ROOT}/camera/object_pred"