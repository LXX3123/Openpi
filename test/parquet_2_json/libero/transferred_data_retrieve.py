import pandas as pd
from PIL import Image
import io
import json
import os # Import the os module

# Define the output directory
output_dir = '/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/libero'
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the image output directory
image_output_dir = os.path.join(output_dir, 'images')
# Create the image directory if it doesn't exist
os.makedirs(image_output_dir, exist_ok=True)

# Load the Parquet file
file_path = '/home/adminroot/lxx/dataset/lerobot_format_spatial/physical-intelligence/libero/data/chunk-000/episode_000000.parquet'
df = pd.read_parquet(file_path)

all_rows_data = []
# To name the output JSON file based on the episode index
output_episode_index_for_filename = None


print(f"Processing all {len(df)} rows from Parquet file: {file_path}")

for index, row in df.iterrows(): # Iterate over the full DataFrame
    # Use new column names from the converted Parquet file
    # Expected columns include: 'image', 'frame_index', 'episode_index'
    try:
        current_episode_idx = row['episode_index']
        current_frame_idx = row['frame_index']
    except KeyError as e:
        print(f"Missing expected column {e} in row {index}. Ensure Parquet file has 'episode_index' and 'frame_index'. Skipping row.")
        continue

    if output_episode_index_for_filename is None:
        output_episode_index_for_filename = current_episode_idx


    print(f"Processing episode_index: {current_episode_idx}, frame_index: {current_frame_idx}")

    image_data = row['image']
    wrist_image_data = row.get('wrist_image') # Get wrist_image, handle if not present
    task_idx = row.get('task_index') # Get task_index
    # log_data_str = row['log'] # REMOVE - 'log' column is not in the new format
    
    image_filepath = None  # Initialize image_filepath
    wrist_image_filepath = None # Initialize wrist_image_filepath

    # --- Process Image ---
    image_bytes = None
    if isinstance(image_data, dict) and 'bytes' in image_data:
        image_bytes = image_data['bytes']
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        print(f"Image data for episode {current_episode_idx}, frame {current_frame_idx} is in an unexpected format: {type(image_data)}")

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            image_filename = f"image_ep{current_episode_idx}_frame_{current_frame_idx}.png" # Use new indices
            image_filepath = os.path.join(image_output_dir, image_filename)
            image.save(image_filepath)
            print(f"Image from episode {current_episode_idx}, frame {current_frame_idx} saved as: {image_filepath}")
        except Exception as e:
            print(f"Could not open or save image for episode {current_episode_idx}, frame {current_frame_idx}: {e}")
            image_filepath = None
    else:
        print(f"No image bytes found for episode {current_episode_idx}, frame {current_frame_idx}.")
        image_filepath = None

    # --- Process Wrist Image ---
    wrist_image_bytes = None
    if isinstance(wrist_image_data, dict) and 'bytes' in wrist_image_data:
        wrist_image_bytes = wrist_image_data['bytes']
    elif isinstance(wrist_image_data, bytes):
        wrist_image_bytes = wrist_image_data
    elif wrist_image_data is not None: # Only print if data exists but is wrong type
        print(f"Wrist image data for episode {current_episode_idx}, frame {current_frame_idx} is in an unexpected format: {type(wrist_image_data)}")

    if wrist_image_bytes:
        try:
            wrist_image = Image.open(io.BytesIO(wrist_image_bytes))
            
            wrist_image_filename = f"wrist_image_ep{current_episode_idx}_frame_{current_frame_idx}.png"
            wrist_image_filepath = os.path.join(image_output_dir, wrist_image_filename)
            wrist_image.save(wrist_image_filepath)
            print(f"Wrist image from episode {current_episode_idx}, frame {current_frame_idx} saved as: {wrist_image_filepath}")
        except Exception as e:
            print(f"Could not open or save wrist image for episode {current_episode_idx}, frame {current_frame_idx}: {e}")
            wrist_image_filepath = None
    elif wrist_image_data is not None: # Only print if data was expected
        print(f"No wrist image bytes found for episode {current_episode_idx}, frame {current_frame_idx}.")
        wrist_image_filepath = None


    # --- Parse Log --- # REMOVE THIS SECTION
    # parsed_log_content = None # REMOVE THIS LINE
    # try: # REMOVE THIS LINE
    #     parsed_log_content = json.loads(log_data_str) # REMOVE THIS LINE
    # except json.JSONDecodeError as e: # REMOVE THIS LINE
    #     print(f"Error decoding JSON from log for row id {row_id}, frame {current_frame_idx}: {e}") # REMOVE THIS LINE
    #     parsed_log_content = log_data_str # REMOVE THIS LINE

    # Prepare data for the current row
    current_row_output_data = {
        "episode_index": int(current_episode_idx), # Use new key
        "frame_index": int(current_frame_idx),   # Use new key
        "image_path": os.path.abspath(image_filepath) if image_filepath else None,
        "wrist_image_path": os.path.abspath(wrist_image_filepath) if wrist_image_filepath else None,
        "task_index": int(task_idx) if task_idx is not None else None,
        # "log": parsed_log_content # REMOVE - 'log' is no longer present
        # Optionally, add other fields from the row if needed:
        "state": row['state'].tolist() if 'state' in row and hasattr(row['state'], 'tolist') else row.get('state'),
        "actions": row['actions'].tolist() if 'actions' in row and hasattr(row['actions'], 'tolist') else row.get('actions'),
        "timestamp": float(row['timestamp']) if 'timestamp' in row else None,
    }
    all_rows_data.append(current_row_output_data)

# Save all collected data to a single JSON file
if all_rows_data:
    # Use the episode_index of the processed data for the output filename
    if output_episode_index_for_filename is not None:
        data_output_filename = f"episode_data_{output_episode_index_for_filename}.json"
    else:
        # Fallback if episode_index couldn't be determined (e.g., empty or malformed parquet)
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        data_output_filename = f"{file_basename}_data.json"
        
    data_output_filepath = os.path.join(output_dir, data_output_filename)
    try:
        with open(data_output_filepath, 'w') as f:
            json.dump(all_rows_data, f, indent=2)
        print(f"\nCollected data for the scene saved to: {data_output_filepath}")
    except Exception as e:
        print(f"Could not save collected data to {data_output_filepath}: {e}")
else:
    print("No data was processed to save.")

# The original section for accessing parts of JSON is commented out below,
# as it depended on 'log_json' being successfully parsed and printed.
# You can adapt it to use 'parsed_log_content' if it's a dictionary.
# if isinstance(parsed_log_content, dict) and 'raw_logs' in parsed_log_content:
# print("\nFirst entry in raw_logs:")
# print(parsed_log_content['raw_logs'][0])