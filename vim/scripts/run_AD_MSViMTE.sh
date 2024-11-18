#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSViMTE_main_SH.py \
-dataset charades \
-mode rgb \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/charades_clip_features_l14 \
-num_clips 256 \
-skip 0 \
--lr 5e-4 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-output_dir workdirs/clip_Charades_test_MSViMTE_e384_vim_d111_23Oct24/