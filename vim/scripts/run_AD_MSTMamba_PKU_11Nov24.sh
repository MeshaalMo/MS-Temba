#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH
   
export CUDA_VISIBLE_DEVICES=1
python AD_MSTMamba_main_PKU_11Nov24.py \
-dataset pku \
-mode rgb \
-backbone clip \
-model hierarchical_vim \
-train True \
-rgb_root /data/msoundar/PKU-MMD-FRAME-CROPPED-Window8-converted/ \
-num_clips 1500 \
-skip 0 \
--lr 0.0005 \
-comp_info False \
-epochs 100 \
-unisize True \
-alpha_l 1 \
-backbone clip \
-beta_l 0.05 \
-batch_size 1 \
-output_dir workdirs/PKU/MSTMamba_clip_PKU_11Nov24/