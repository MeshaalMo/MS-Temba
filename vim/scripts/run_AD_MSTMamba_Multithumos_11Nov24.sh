#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSTMamba_main_10Nov24.py \
-dataset multithumos \
-mode rgb \
-backbone i3d \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/multithumos_features_i3d/ \
-num_clips 800 \
-skip 0 \
--lr 5e-4 \
-comp_info False \
-epochs 100 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 8 \
-output_dir workdirs/multithumos/MSTMamba_dilssm_adfus_mbaint_p800_i3d_MT_11Nov24/