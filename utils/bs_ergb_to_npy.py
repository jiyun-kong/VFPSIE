import argparse
import os
import numpy as np
import glob
import shutil

FRAME_WIDTH = 970
FRAME_HEIGHT = 625


def convert_and_fix_event_pixels(data, upper_limit, fix_overflows=True):
    data = data.astype(np.int32)
    overflow_indices = np.where(data > upper_limit*32)
    num_overflows = overflow_indices[0].shape[0]

    if fix_overflows and num_overflows > 0:
        data[overflow_indices] = data[overflow_indices] - 65536
        
    data = data / 32.0
    data = np.rint(data)
    data = data.astype(np.int16)
    data = np.clip(data, 0, upper_limit)
    return data


def bs_ergb_to_npy(input_path, output_path):
    events_folder = os.path.join(input_path, 'events')
    event_files_glob_pattern = os.path.join(events_folder, "*.npz")
    event_file_paths = sorted(glob.glob(event_files_glob_pattern))
    
    os.makedirs(output_path, exist_ok=True)
    
    for event_file_path in event_file_paths:
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(event_file_path))[0]
        output_file = os.path.join(output_path, f"{base_filename}.npy")
        
        print(f"Processing file: {base_filename}")
        
        # Load and process events
        event_file_data = np.load(event_file_path)
        num_events = event_file_data['x'].shape[0]
        
        # Initialize array for events
        events_data = np.zeros((num_events, 4), dtype=np.float32)  # [t, x, y, p]
        
        # Convert and fix coordinates
        x_data = convert_and_fix_event_pixels(event_file_data['x'], FRAME_WIDTH - 1)
        y_data = convert_and_fix_event_pixels(event_file_data['y'], FRAME_HEIGHT - 1)
        
        # Store events in [t, x, y, p] format
        events_data[:, 0] = event_file_data['timestamp'] / 1000000.0  # convert us to s
        events_data[:, 1] = x_data
        events_data[:, 2] = y_data
        events_data[:, 3] = event_file_data['polarity']
        
        # Save the events data
        np.save(output_file, events_data)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", help="path to the base directory containing sequence folders")
    args = parser.parse_args()

    # Get all sequence folders
    sequence_folders = [f for f in os.listdir(args.base_path) 
                       if os.path.isdir(os.path.join(args.base_path, f))]

    for seq_folder in sequence_folders:
        input_path = os.path.join(args.base_path, seq_folder)
        output_path = os.path.join(input_path, 'events_npy')
        
        print(f"\nProcessing sequence: {seq_folder}")
        bs_ergb_to_npy(input_path, output_path)
        print(f"Completed processing sequence: {seq_folder}")
