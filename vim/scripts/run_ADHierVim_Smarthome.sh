#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python AD_hiermamba_main_SH.py \
-dataset tsu \
-mode rgb \
-model hierarchical_vim \
-train True \
-rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_i3d/ \
-num_clips 2000 \
-skip 0 \
--lr 5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 32 \
-output_dir workdirs/I3D_SH_nomamba_mstctfusion_e384_vim_d111_23Oct24/