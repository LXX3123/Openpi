import pandas as pd
from PIL import Image
import io
import json
import os # Import the os module

# Define the output directory
output_dir = '/home/adminroot/lxx/openpi/code/openpi/test/parquet_2_json/uav_flow'
# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the image output directory
image_output_dir = os.path.join(output_dir, 'images')
# Create the image directory if it doesn't exist
os.makedirs(image_output_dir, exist_ok=True)

# Load the Parquet file
file_path = '/home/adminroot/lxx/dataset/uav_flow/categorized/retreat/retreat-00006.parquet'
df = pd.read_parquet(file_path)

# Sort the DataFrame by 'id' and then 'frame_idx'
if not df.empty:
    df.sort_values(by=['id', 'frame_idx'], inplace=True)

# List to store data for all processed rows
all_rows_data = []
# To name the output JSON file based on the first processed row's ID
first_processed_row_id = None
first_processed_frame_idx = None

# 获取第一个出现的唯一ID
first_id = df['id'].iloc[0]
print(f"处理ID为 {first_id} 的数据")

# 仅筛选该ID的所有行并重置索引
id_df = df[df['id'] == first_id].reset_index(drop=True)
print(f"找到 {len(id_df)} 行数据，ID为 {first_id}")

# 设置输出文件名变量
first_processed_row_id = first_id
first_processed_frame_idx = id_df['frame_idx'].iloc[0]

# 处理该ID的所有行（替换当前的for循环）
for index, row in id_df.iterrows():
    row_id = row['id']
    current_frame_idx = row['frame_idx']

    if first_processed_row_id is None: # Capture the first row's id and frame_idx
        first_processed_row_id = row_id
        first_processed_frame_idx = current_frame_idx

    # Stop if frame_idx resets to 0, but only after processing at least one row
    # and if the current_frame_idx is 0 and it's not the very first frame_idx we are processing (which could be 0)
    if current_frame_idx == 0 and index > 0 and id_df.iloc[index-1]['frame_idx'] != 0 :
        print(f"frame_idx reset to 0 at row {index}. Stopping processing for this scene.")
        break

    print(f"Processing row with id: {row_id}, frame_idx: {current_frame_idx}")

    image_data = row['image']
    log_data_str = row['log']
    
    image_filepath = None  # Initialize image_filepath

    # --- Process Image ---
    image_bytes = None
    if isinstance(image_data, dict) and 'bytes' in image_data:
        image_bytes = image_data['bytes']
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        print(f"Image data for row id {row_id} is in an unexpected format: {type(image_data)}")

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            image_filename = f"image_{row_id}_frame_{current_frame_idx}.png"
            image_filepath = os.path.join(image_output_dir, image_filename) # Changed to image_output_dir
            image.save(image_filepath)
            print(f"Image from row id {row_id} saved as: {image_filepath}")
            # image.show() # Optional: display each image
        except Exception as e:
            print(f"Could not open or save image for row id {row_id}: {e}")
            image_filepath = None # Ensure image_filepath is None if saving failed
    else:
        print(f"No image bytes found for row id {row_id}.")
        image_filepath = None

    # --- Parse Log ---
    parsed_log_content = None
    try:
        parsed_log_content = json.loads(log_data_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from log for row id {row_id}, frame {current_frame_idx}: {e}")
        parsed_log_content = log_data_str

    # Prepare data for the current row
    current_row_output_data = {
        "id": row_id,
        "frame_idx": int(current_frame_idx),
        "image": os.path.abspath(image_filepath) if image_filepath else None,
        "log": parsed_log_content
    }
    all_rows_data.append(current_row_output_data)

# Save all collected data to a single JSON file
if all_rows_data:
    # Use the ID and frame_idx of the first processed row for the output filename
    data_output_filename = f"scene_data_start_id_{first_processed_row_id}_frame_{first_processed_frame_idx}.json"
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