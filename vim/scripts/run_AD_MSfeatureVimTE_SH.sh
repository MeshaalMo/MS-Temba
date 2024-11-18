#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSfeatureViMTE_main_v2.py \
-dataset tsu \
-mode rgb \
-backbone i3d \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_i3d/ \
-num_clips 2500 \
-skip 0 \
--lr 4.5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 2 \
-output_dir workdirs/SOTA_MSTMamba_intrablk_Mamba_i3d_SH_10Nov24/