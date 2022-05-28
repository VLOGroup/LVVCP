#! /bin/bash

echo "Run this demo from the root folder of the repository ./demo_scripts/<script_name.sh>"
# export CUDA_VISIBLE_DEVICES=1

python demo_LVVCP.py \
    --cfg_strict_reload=1 \
    --out_path=out_val/ \
    --load_path=data_and_model_samples/DAVIS2017_scootergray/res_mm1_ep400_CUDA_Op_sample_cv/model_NHWC_bilin/mm1__ep400_3Cands12ItersModel_NHWC_p2.ckpt \
    --list_val=data_and_model_samples/DAVIS2017_scootergray/scootergray.txt \
    --dataset_path=data_and_model_samples/ \
    --minimal_output=1 \
    --filesaver_type=MP 

printf "\n"
echo "For a numerical quick check you can run:"
echo "  python numerical_quick_check.py --sample_cv=1 --res_path=out_val/<results_folder_name>/"
printf "\n\n"