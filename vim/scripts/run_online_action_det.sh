#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python OAD_sliding_window.py \
--checkpoint /data/asinha13/projects/MAD/Vim/vim/workdirs/clip_SH_MSfeatureVimTE_dilatedTCdilatedViM_3B_tokgatefuse_mambaintrctn_d111_31Oct24/best_model.pth \
--output_dir /data/asinha13/projects/MAD/Vim/vim/workdirs/OAD/clip_TSU_gated_cumulative_sliding_window8_3Nov24 \
--rgb_root /data/asinha13/projects/MAD/MS-TCT/data/smarthome_features_clip \
--batch_size 1 \
--window_size 8 \
--dataset tsu

python convert_pkl.py \
--input_pkl /data/asinha13/projects/MAD/Vim/vim/workdirs/OAD/clip_TSU_gated_cumulative_sliding_window8_3Nov24/sliding_window_predictions.pkl \
--output_pkl /data/asinha13/projects/MAD/Vim/vim/workdirs/OAD/clip_TSU_gated_cumulative_sliding_window8_3Nov24/converted_sliding_window_predictions.pkl

python Evaluation_v2.py \
-pkl_path /data/asinha13/projects/MAD/Vim/vim/workdirs/OAD/clip_TSU_gated_cumulative_sliding_window8_3Nov24/converted_sliding_window_predictions.pkl \
-data tsu