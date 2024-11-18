#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_main.py \
-dataset charades \
-mode rgb \
-model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2 \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/charades_i3d_features/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-output_dir workdirs/vimB_AD_Charades/