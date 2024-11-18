#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_MSfeatureViMTE_main_v2_test.py \
--checkpoint /data/asinha13/projects/MAD/Vim/vim/workdirs/tsu/SH_MSfeatureVimTE_dilatedTCdilatedViM_3B_addfusion_mambainteraction_d111_28Oct24/best_model.pth \
--eval_only \
-output_dir workdirs/test \
-dataset tsu \
-mode rgb \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_i3d/ \
-num_clips 2500 \
-skip 0 \
--lr 5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 2 \