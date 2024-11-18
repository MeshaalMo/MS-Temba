#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSfeatureViMTE_main_v2.py \
-dataset charades \
-mode rgb \
-model hierarchical_vim_1block \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/charades_i3d_features/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 8 \
-output_dir workdirs/SOTA_MSTMamba_intrablk_Mamba_i3d_CH_10Nov24/