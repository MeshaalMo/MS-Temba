import pickle
import numpy as np
import os

def load_predictions(pkl_path):
    """Load predictions from a .pkl file."""
    with open(pkl_path, 'rb') as f:
        predictions = pickle.load(f)
    return predictions

def get_action_intervals(logits, threshold=0.5, window_size=16):
    """
    Process logits to find action intervals in terms of frames.
    
    Args:
    - logits: numpy array of shape (C, T), where C is number of classes, T is number of segments.
    - threshold: float, the score threshold to consider an action as performed.
    - window_size: int, the number of frames each segment represents.
    
    Returns:
    - action_intervals: dict, action intervals for each action with frame start and end.
    """
    num_classes, num_segments = logits.shape
    action_intervals = {}

    for class_idx in range(num_classes):
        active_segments = np.where(logits[class_idx] >= threshold)[0]

        if active_segments.size > 0:
            # Initialize start of the first interval
            start_frame = active_segments[0] * window_size
            prev_segment = active_segments[0]
            
            for segment in active_segments[1:]:
                # Check if the segment is contiguous; if not, finalize the current interval
                if segment != prev_segment + 1:
                    end_frame = (prev_segment + 1) * window_size - 1  # end of the interval
                    action_intervals.setdefault(class_idx, []).append((start_frame, end_frame))
                    start_frame = segment * window_size  # start a new interval
                prev_segment = segment
            
            # Add the final interval
            end_frame = (prev_segment + 1) * window_size - 1
            action_intervals.setdefault(class_idx, []).append((start_frame, end_frame))

    return action_intervals

def process_pkl_file(pkl_path, output_file, threshold=0.5, window_size=16):
    """
    Process a .pkl file to get action intervals for each video and save to a file.
    
    Args:
    - pkl_path: str, path to the .pkl file with predictions.
    - output_file: str, path to the output file where results are saved.
    - threshold: float, the score threshold to consider an action as performed.
    - window_size: int, the number of frames each segment represents.
    """
    predictions = load_predictions(pkl_path)
    with open(output_file, 'w') as f:
        for video_id, logits in predictions.items():
            #f.write(f"Video ID: {video_id}\n")
            action_intervals = get_action_intervals(np.array(logits), threshold, window_size)
            
            for action_idx, intervals in action_intervals.items():
                for interval in intervals:
                    start_frame, end_frame = interval
                    f.write(f"{action_idx} {start_frame} {end_frame} {video_id}\n")
            f.write("\n")

# Usage
# Replace 'your_predictions.pkl' with the path to your actual .pkl file
# Replace 'output.txt' with the desired path for the output file
pkl_path = './save_logit/44.pkl'
output_file = 'output.txt'
process_pkl_file(pkl_path, output_file, threshold=0.5, window_size=16)

