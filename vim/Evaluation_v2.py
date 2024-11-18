import json
import pickle5 as pickle
from apmeter import APMeter
import numpy as np
from utils import *
import warnings
import argparse
import os
import logging
from datetime import datetime
import csv
import time

warnings.filterwarnings('ignore')

def setup_logging(pkl_path):
    log_dir = os.path.dirname(pkl_path)
    log_filename = f"evaluation_log_{datetime.now().strftime('%d%b%y').lower()}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Logging to: {log_path}")
    return log_dir

def save_aps(aps, log_dir):
    ap_filename = f"class_aps_{datetime.now().strftime('%d%b%y').lower()}.csv"
    ap_path = os.path.join(log_dir, ap_filename)
    
    with open(ap_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'AP'])
        for i, ap in enumerate(aps):
            # Detach tensor and convert to float
            ap_value = float(ap.detach().cpu().numpy())
            writer.writerow([i, ap_value])
    
    logging.info(f"Saved class APs to: {ap_path}")

def make_gt(gt_file, logits, num_classes=157):
    start_time = time.time()
    gt_new = {}
    vid_length = {}
    fps_seg = {}
    with open(gt_file, 'r') as f:
        gt = json.load(f)
    #print(dict(list(gt.items())[0:2]))
    i = 0
    gt_len=0
    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue
        else:
            gt_len=gt_len+1
    for vid in gt.keys():
        if gt[vid]['subset'] != "testing":
            continue
        if vid not in logits.keys():
            continue
        num_pred = logits[vid].shape[1]
        print(logits[vid].shape)

        label = np.zeros((num_pred, num_classes), np.float32)

        fps = float(num_pred / float(gt[vid]['duration']))
        for ann in gt[vid]['actions']:
            for fr in range(0, num_pred, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1
        gt_new[vid]=label
        vid_length[vid]=gt[vid]['duration']
        fps_seg[vid]=fps
        i += 1
    end_time = time.time()
    logging.info(f"Time taken to process ground truth: {end_time - start_time:.2f} seconds")
    return gt_new, vid_length, fps_seg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-pkl_path', type=str)  # './test.pkl'
    parser.add_argument('-data', type=str, default='charades')  # './test.pkl'
    args = parser.parse_args()

    pkl_path = args.pkl_path
    log_dir = setup_logging(pkl_path)

    logging.info(f"Starting evaluation with pickle file: {pkl_path}")

    if args.data == 'charades':
        gt_file = '/data/asinha13/projects/MAD/MS-TCT/data/charades.json'
        classes = 157
    elif args.data == 'tsu':
        gt_file = '/data/asinha13/projects/MAD/MS-TCT/data/smarthome_CS_51.json'
        classes = 51
    elif args.data == 'multithumos':
        gt_file = '/data/asinha13/projects/MAD/MS-TCT/data/modified_multithumos.json'
        classes = 65

    logging.info(f"Ground truth file: {gt_file}")
    logging.info(f"Number of classes: {classes}")

    start_time = time.time()
    with open(pkl_path, 'rb') as pkl:
        logits = pickle.load(pkl)
    end_time = time.time()
    logging.info(f"Time taken to load logits: {end_time - start_time:.2f} seconds")
    logging.info(f"Loaded logits from pickle file. Number of videos: {len(logits)}")

    start_time = time.time()
    gt_new, vid_len, fps_seg = make_gt(gt_file, logits, classes)
    end_time = time.time()
    logging.info(f"Time taken to process ground truth: {end_time - start_time:.2f} seconds")

    # Compute mAP
    start_time = time.time()
    apm = APMeter()
    sampled_apm = APMeter()
    first_idx = 0
    idx = 0
    pred_probs = []
    gt_labels = []
    total_frames = 0

    for vid in gt_new.keys():
        idx += 1
        logit = np.transpose(logits[vid], (1, 0))
        total_frames += logit.shape[0]

        apm.add(logit, gt_new[vid])
        sampled_25_inference(logit, gt_new[vid], sampled_apm)
        pred_probs.append(logit)
        gt_labels.append(gt_new[vid])

    end_time = time.time()
    evaluation_time = end_time - start_time
    logging.info(f"Time taken for evaluation: {evaluation_time:.2f} seconds")
    
    throughput = total_frames / evaluation_time
    logging.info(f"Throughput: {throughput:.2f} frames/second")

    # per-frame mAP
    val_map = 100 * apm.value().mean()
    sample_val_map = 100 * sampled_apm.value().mean()
    logging.info(f"Test Frame-based mAP: {val_map:.2f}")
    logging.info(f"25 sampled Frame-based mAP: {sample_val_map:.2f}")
    
    # Save APs for each class
    class_aps = 100 * apm.value()
    save_aps(class_aps, log_dir)
    logging.info(f"APs for the classes: {class_aps.detach().cpu().numpy()}")
    logging.info(f"APs for the classes: {class_aps}")

    # action-conditional metrics for different t
    start_time = time.time()
    for t in [0, 20, 40]:
        prec, re, ns, map_val = conditional_metric(pred_probs, gt_labels, t=t, avg=True)
        fs = get_f1(prec, re)  # action conditional f1-score
        logging.info(f"Metrics for t={t}:")
        logging.info(f"  Precision(c_i|c_j,{t})= {prec:.4f}")
        logging.info(f"  Recall(c_i|c_j,{t})= {re:.4f}")
        logging.info(f"  F1Score(c_i|c_j,{t})= {fs:.4f}")
        logging.info(f"  mAP(c_i|c_j,{t})= {map_val:.4f}")
    end_time = time.time()
    logging.info(f"Time taken for conditional metrics: {end_time - start_time:.2f} seconds")

    logging.info("Evaluation completed.")


