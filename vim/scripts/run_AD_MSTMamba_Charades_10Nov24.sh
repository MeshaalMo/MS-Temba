#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSTMamba_main_10Nov24_v2.py \
-dataset charades \
-mode rgb \
-backbone clip \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/charades_clip_features_l14/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-output_dir workdirs/abl_MSTMamba_4BLK_clip_CH_14Nov24/