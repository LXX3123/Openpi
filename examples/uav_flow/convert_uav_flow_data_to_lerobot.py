"""
Minimal example script for converting a custom UAV-Flow dataset (from Parquet files) to LeRobot format.

The script reads Parquet files from the specified data directory, processes them episode by episode
(grouped by 'id'), and converts them into LeRobot's format.

Usage:
uv run examples/UAV-FLOW/convert_uav_flow_data_to_lerobot.py --data_dir /path/to/your/uav_parquet_files_folder

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/UAV-FLOW/convert_uav_flow_data_to_lerobot.py --data_dir /path/to/your/uav_parquet_files_folder --push_to_hub

Note: to run the script, you need to install pandas, pyarrow, and Pillow:
`uv pip install pandas pyarrow Pillow`

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import shutil
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import io 
import json 
import logging 
import gc # Import the garbage collection module
import psutil
import traceback
import sys

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

# Configuration for the new dataset
REPO_NAME = "uav_flow"  # Name of the output dataset, also used for the Hugging Face Hub. You can change this.
TARGET_IMAGE_HEIGHT = 256  # Target height for images after resizing. Adjust as needed.
TARGET_IMAGE_WIDTH = 256   # Target width for images after resizing. Adjust as needed.
FPS = 5          # 原文中提到uniformly sample the video at 5 Hz
MAX_EPISODES_TO_PROCESS = 3000 # Set to a small number for testing, or None to process all episodes
PARQUET_FILE_PREFIX = "pass" # Prefix for the parquet files to be processed.


def process_parquet_file(pq_file, dataset, logger, processed_episodes, episodes_processed_count, MAX_EPISODES_TO_PROCESS):
    """
    Processes a single Parquet file, extracts episodes, and adds them to the LeRobot dataset.
    Returns the updated count of processed episodes and a flag indicating if any episode was saved.
    """
    any_episode_saved_from_file = False
    try:
        # Force garbage collection before loading new file
        gc.collect()
        current_df = pd.read_parquet(pq_file)
        logger.debug(f"File {pq_file} loaded with {len(current_df)} rows.")
    except Exception as e:
        logger.error(f"Error reading Parquet file {pq_file}: {e}. Skipping this file.")
        return episodes_processed_count, any_episode_saved_from_file

    if current_df.empty:
        logger.warning(f"File {pq_file} is empty. Skipping.")
        return episodes_processed_count, any_episode_saved_from_file

    # Sort by id and frame_idx to ensure correct order within episodes for this file.
    current_df.sort_values(by=['id', 'frame_idx'], inplace=True)

    # Group by 'id' to process each episode within the current file
    episode_groups = list(current_df.groupby('id'))

    for episode_id, episode_df in episode_groups:
        if str(episode_id) in processed_episodes:
            logger.debug(f"Skipping already processed episode '{episode_id}'.")
            continue

        if MAX_EPISODES_TO_PROCESS is not None and episodes_processed_count >= MAX_EPISODES_TO_PROCESS:
            logger.info(f"Reached MAX_EPISODES_TO_PROCESS limit ({MAX_EPISODES_TO_PROCESS}) while processing episodes in {pq_file}. Stopping.")
            break

        episode_objects = episode_df.to_dict('records')
        num_objects_in_episode = len(episode_objects)

        logger.debug(f"Processing episode '{episode_id}' with {num_objects_in_episode} objects.")

        if num_objects_in_episode < 2:
            logger.warning(f"Episode '{episode_id}' has {num_objects_in_episode} objects, less than the required 2. Skipping.")
            continue

        try:
            raw_common_log = episode_objects[0]['log']
            common_log = json.loads(raw_common_log) if isinstance(raw_common_log, str) else raw_common_log
            if not isinstance(common_log, dict):
                logger.error(f"Episode '{episode_id}': 'log' field is not a dictionary or valid JSON string. Skipping.")
                continue
            
            task_instruction = common_log.get('instruction_unified', "")
            preprocessed_logs_list = common_log.get('preprocessed_logs')

            if preprocessed_logs_list is None:
                logger.error(f"Episode '{episode_id}': 'preprocessed_logs' missing in 'log' data. Skipping.")
                continue
            
            if len(preprocessed_logs_list) != num_objects_in_episode:
                logger.error(f"Episode '{episode_id}': Mismatch in log list length and number of objects. Skipping.")
                continue
        except (KeyError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Episode '{episode_id}': Error processing 'log' data: {e}. Skipping episode.")
            continue

        try:
            raw_first_image_data = episode_objects[0]['image']
            first_image_bytes = raw_first_image_data.get('bytes') if isinstance(raw_first_image_data, dict) else raw_first_image_data
            if not isinstance(first_image_bytes, bytes) or not first_image_bytes:
                logger.error(f"Episode '{episode_id}': First image data is not valid. Skipping episode.")
                continue
            pil_first_image = Image.open(io.BytesIO(first_image_bytes)).convert("RGB").resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
            first_image_arr = np.array(pil_first_image, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Episode '{episode_id}': Error processing first image: {e}. Skipping episode.")
            continue
        
        frames_added_to_this_episode = 0
        for step_idx in range(num_objects_in_episode - 1):
            try:
                raw_current_image_data = episode_objects[step_idx]['image']
                current_image_bytes = raw_current_image_data.get('bytes') if isinstance(raw_current_image_data, dict) else raw_current_image_data
                if not isinstance(current_image_bytes, bytes) or not current_image_bytes:
                    logger.warning(f"Episode '{episode_id}', Step {step_idx}: Wrist image data is not valid. Skipping step.")
                    continue
                pil_wrist_image = Image.open(io.BytesIO(current_image_bytes)).convert("RGB").resize((TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
                wrist_image_arr = np.array(pil_wrist_image, dtype=np.uint8)

                state_data = np.array(preprocessed_logs_list[step_idx], dtype=np.float32)
                action_data = np.array(preprocessed_logs_list[step_idx + 1], dtype=np.float32)
                if state_data.shape != (6,) or action_data.shape != (6,):
                    logger.warning(f"Episode '{episode_id}', Step {step_idx}: Incorrect state/action shape. Skipping step.")
                    continue

                frame_payload = {
                    "image": first_image_arr,
                    "wrist_image": wrist_image_arr,
                    "state": state_data,
                    "actions": action_data,
                    "task": task_instruction,
                }
                dataset.add_frame(frame_payload)
                frames_added_to_this_episode += 1
            except Exception as e:
                logger.warning(f"Episode '{episode_id}', Step {step_idx}: Error processing step: {e}. Skipping step.")
                continue

        if frames_added_to_this_episode > 0:
            dataset.save_episode()
            logger.debug(f"Successfully saved episode '{episode_id}' with {frames_added_to_this_episode} steps.")
            episodes_processed_count += 1
            processed_episodes[str(episode_id)] = task_instruction
            logger.info(f"Total episodes processed: {episodes_processed_count}")
            any_episode_saved_from_file = True
        else:
            logger.warning(f"No frames were added for episode '{episode_id}', so it was not saved.")

        if episodes_processed_count % 10 == 0:
            gc.collect()

    # Explicitly clean up to be safe
    del episode_groups
    del current_df
    gc.collect()
    
    return episodes_processed_count, any_episode_saved_from_file


def main(data_dir: str, *, push_to_hub: bool = False):
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) # Set base level for the logger to capture all desired levels

    # Clear existing handlers if any (e.g., during re-runs in an interactive session)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler for error logs
    error_log_path = "/home/adminroot/lxx/openpi/code/openpi/examples/uav_flow/error.log"
    try:
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        file_handler = logging.FileHandler(error_log_path, mode='w') # 'w' to overwrite log file on each run
    except OSError as e:
        print(f"Warning: Could not create directory for error log file {error_log_path}: {e}. Errors will not be saved to file.")
        file_handler = None

    if file_handler:
        file_handler.setLevel(logging.WARNING)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Create console handler for non-error logs (INFO, WARNING)
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO) # Log INFO and WARNING to console
    # Filter out WANRING and ERROR and CRITICAL for console handler
    # console_handler.addFilter(lambda record: record.levelno < logging.WARNING) # Temporarily disable or modify this filter
    console_handler.setLevel(logging.INFO) # Show INFO and WARNING logs on console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # More detailed format
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    try:
        # Clean up any existing dataset in the output directory
        output_path = HF_LEROBOT_HOME / REPO_NAME
        if output_path.exists():
            logger.info(f"Cleaning up existing dataset at {output_path}")
            # shutil.rmtree(output_path) # Consider commenting this out if you want to debug continuation
        # output_path.mkdir(parents=True, exist_ok=True) # Remove this line

        # Create LeRobot dataset, define features to store
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="uav",  # Set robot type to UAV
            fps=FPS,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (6,),  # 6-dim UAV state from preprocessed_logs
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (6,),  # 6-dim UAV action (next state from preprocessed_logs)
                    "names": ["actions"],
                },
                # "timestamp": { # Added timestamp feature
                #     "dtype": "float32", # Timestamps can be large and require precision
                #     "shape": (), # MODIFIED: Changed from (1,) to () to represent a scalar
                #     "names": ["timestamp"],
                # }
            },
            image_writer_threads=10,  # Default, adjust if needed
            image_writer_processes=5, # Default, adjust if needed
        )

        # Find and load all Parquet files
        file_pattern = f"{PARQUET_FILE_PREFIX}-*.parquet"
        parquet_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if not parquet_files:
            logger.error(f"No Parquet files found in '{data_dir}' matching '{file_pattern}'. Please check the path.")
            return

        logger.info(f"Found {len(parquet_files)} Parquet files to process.")
        
        episodes_processed_count = 0
        any_episode_processed_successfully = False # To track if at least one episode was saved

        # Add checkpoint functionality by tracking processed episodes
        checkpoint_file = os.path.join(os.path.dirname(error_log_path), "conversion_checkpoint.txt")
        processed_episodes = {}
        last_processed_file_idx = 0

        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    checkpoint_data = f.read().strip().split("\n")
                    if len(checkpoint_data) >= 1 and checkpoint_data[0]:
                        last_processed_file_idx = int(checkpoint_data[0])
                    if len(checkpoint_data) > 1:
                        for line in filter(None, checkpoint_data[1:]):
                            parts = line.split(';', 1)
                            if parts:
                                episode_id = parts[0]
                                instruction = parts[1] if len(parts) > 1 else ""
                                processed_episodes[episode_id] = instruction
                logger.info(f"Loaded checkpoint: starting from file index {last_processed_file_idx}, {len(processed_episodes)} episodes already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")

        # Process each Parquet file individually
        for pq_file_idx, pq_file in enumerate(parquet_files):
            if pq_file_idx < last_processed_file_idx:
                logger.info(f"Skipping already processed file {pq_file_idx + 1}/{len(parquet_files)}: {pq_file}")
                continue

            if MAX_EPISODES_TO_PROCESS is not None and episodes_processed_count >= MAX_EPISODES_TO_PROCESS:
                logger.info(f"Reached MAX_EPISODES_TO_PROCESS limit ({MAX_EPISODES_TO_PROCESS}) before processing file {pq_file_idx + 1}/{len(parquet_files)}. Stopping.")
                break

            logger.info(f"Processing file {pq_file_idx + 1}/{len(parquet_files)}: {pq_file}...")
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory before file {pq_file_idx + 1}: {mem_before:.2f}MB")

            episodes_processed_count, saved_any = process_parquet_file(
                pq_file, dataset, logger, processed_episodes, episodes_processed_count, MAX_EPISODES_TO_PROCESS
            )
            if saved_any:
                any_episode_processed_successfully = True
            
            # Force garbage collection and log memory usage after each file
            gc.collect()
            
            mem_after = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory after file {pq_file_idx + 1}: {mem_after:.2f}MB (change: {mem_after - mem_before:+.2f}MB)")
            
            # Optional: If memory usage is too high, force more aggressive cleanup
            if mem_after > 32000:  # If memory exceeds 32GB
                logger.warning(f"High memory usage detected: {mem_after:.2f}MB. Forcing aggressive cleanup.")
                # Force Python to return memory to OS
                import ctypes
                if hasattr(ctypes, 'pythonapi') and hasattr(ctypes.pythonapi, 'PyGC_Collect'):
                    ctypes.pythonapi.PyGC_Collect()
            
            # After finishing a file, save checkpoint for the *next* file
            with open(checkpoint_file, "w") as f:
                f.write(f"{pq_file_idx + 1}\n")
                for ep_id, instruction in processed_episodes.items():
                    f.write(f"{ep_id};{instruction}\n")

        logger.info("Dataset conversion process finished.")
        if not any_episode_processed_successfully:
            logger.warning("No episodes were successfully processed and saved from any of the files.")
        else:
            logger.info(f"Successfully processed and saved {episodes_processed_count} episode(s) in total.")
            if MAX_EPISODES_TO_PROCESS is not None and episodes_processed_count < MAX_EPISODES_TO_PROCESS:
                logger.info(f"Note: The MAX_EPISODES_TO_PROCESS limit was {MAX_EPISODES_TO_PROCESS}, but only {episodes_processed_count} were processed. This could be due to reaching the end of available data or episodes being skipped due to errors/filters.")


        # Consolidate the dataset (optional, can be done later)
        # logger.info("Consolidating dataset...")
        # dataset.consolidate(run_compute_stats=False) # Set run_compute_stats=True if you want stats computed now

        # Optionally push to the Hugging Face Hub
        if push_to_hub:
            logger.info(f"Pushing dataset {REPO_NAME} to Hugging Face Hub...")
            dataset.push_to_hub(
                repo_id=REPO_NAME, # Ensure repo_id is passed if it's different from the one in create() or for clarity
                tags=["uav-flow", "uav", "custom-dataset"], # Add relevant tags
                private=False,  # Set to True if you want a private dataset on the Hub
                push_videos=True, # If you want to generate and push video previews
                license="apache-2.0", # Or your preferred license
            )
            logger.info("Dataset pushed to Hub.")

    except Exception as e:
        logger.error("Unhandled exception occurred in main:", exc_info=True)
        # Write detailed exception info to a separate file for thorough debugging
        with open(os.path.join(os.path.dirname(error_log_path), "crash_details.txt"), "w") as crash_file:
            crash_file.write(f"Exception: {str(e)}\n\n")
            crash_file.write(f"Stack trace:\n{traceback.format_exc()}\n\n")
            crash_file.write(f"Python version: {sys.version}\n")
            crash_file.write(f"Memory usage at crash: RSS={psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f}MB\n")
        raise  # Re-raise to ensure proper exit code
    finally:
        logging.shutdown() # Ensure all handlers are flushed


if __name__ == "__main__":
    tyro.cli(main)
