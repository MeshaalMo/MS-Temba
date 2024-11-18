#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSTMamba_main_10Nov24_v2.py \
-dataset tsu \
-mode rgb \
-backbone clip \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_clip/ \
-num_clips 2500 \
-skip 0 \
--lr 4.5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1 \
-output_dir workdirs/abl_MSTMamba_4BLK_clip_SH_14Nov24/