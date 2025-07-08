import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
from PIL import Image
import os # Added for directory creation
import json # Import the json module

# 假设你的 Parquet 文件名为 'your_dataset.parquet'
file_path = '/home/adminroot/lxx/dataset/uav_flow_lerobot_format/uav_flow/data/chunk-000/episode_000002.parquet'
# file_path = '/home/adminroot/lxx/dataset/lerobot_format_spatial/physical-intelligence/libero/data/chunk-000/episode_000000.parquet'

# 方法一：使用 pandas 读取文件并查看信息
try:
    df = pd.read_parquet(file_path)

    print("=== DataFrame Info ===")
    df.info()  # 查看列名、数据类型、非空值数量

    # Create output directory if it doesn't exist
    output_image_dir = "/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/uav_flow/images"
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"\nImages will be saved to: {os.path.abspath(output_image_dir)}")

    print("\n\n=== Data for First 1 Rows ===")
    num_rows_to_process = min(1, len(df)) # Process up to 1 rows, or fewer if df is smaller

    for i in range(num_rows_to_process):
        print(f"\n\n=== Row {i} Data ===")
        current_row = df.iloc[i]
        for col_name in df.columns:
            print(f"\n--- Column: {col_name} (Row: {i}) ---")
            col_value = current_row[col_name]

            if col_name == 'image' or col_name == 'wrist_image':
                try:
                    if isinstance(col_value, dict) and 'bytes' in col_value:
                        image_bytes = col_value['bytes']
                        if image_bytes: # Check if bytes are not empty
                            print("11111111")
                            image = Image.open(BytesIO(image_bytes))
                            print(f"{col_name.capitalize()} format: {image.format}, size: {image.size}, mode: {image.mode}")
                            image_filename = os.path.join(output_image_dir, f"row_{i}_{col_name}.png")
                            image.save(image_filename)
                            print(f"{col_name.capitalize()} saved to {image_filename}")
                        else:
                            print(f"{col_name.capitalize()} data: No image bytes found in dictionary.")
                    elif isinstance(col_value, bytes): # Handle if it's directly bytes (less common for Parquet dicts)
                        print("222222222")
                        image_bytes = col_value
                        if image_bytes:
                            image = Image.open(BytesIO(image_bytes))
                            print(f"{col_name.capitalize()} format: {image.format}, size: {image.size}, mode: {image.mode}")
                            image_filename = os.path.join(output_image_dir, f"row_{i}_{col_name}.png")
                            image.save(image_filename)
                            print(f"{col_name.capitalize()} saved to {image_filename}")
                        else:
                            print(f"{col_name.capitalize()} data: No image bytes found.")
                    else:
                        print("333333")
                        print(f"{col_name.capitalize()} data: {str(col_value)[:10]} (Type: {type(col_value)})") # Print limited string representation
                except Exception as e:
                    print(f"Error processing {col_name} for row {i}: {e}. Value was: {str(col_value)[:100]}")
            elif col_name == 'log':
                print(f"Data type of 'log' column entry: {type(col_value)}")
                log_data_to_inspect = None
                if isinstance(col_value, str):
                    try:
                        log_data_to_inspect = json.loads(col_value)
                        print("Successfully parsed 'log' string as JSON.")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from 'log' string: {e}")
                        print(f"'log' string content (first 100 chars): {col_value[:100]}")
                elif isinstance(col_value, dict):
                    log_data_to_inspect = col_value # Already a dict

                if isinstance(log_data_to_inspect, dict):
                    print("Structure of the 'log' entry:")
                    for log_key, log_item_value in log_data_to_inspect.items():
                        print(f"  - Key: '{log_key}', Value Type: {type(log_item_value)}")
                        if isinstance(log_item_value, list) and len(log_item_value) > 0:
                            first_element = log_item_value[0]
                            print(f"    - '{log_key}' is a list. First element type: {type(first_element)}")
                            if isinstance(first_element, list) and len(first_element) > 0:
                                inner_first_element = first_element[0]
                                print(f"      - Inner list's first element type: {type(inner_first_element)}")
                        # Not printing content of log items as requested
                elif log_data_to_inspect is None and not isinstance(col_value, str): # If it wasn't a string and not a dict
                    print(f"'log' content (first 100 chars if not a dict/list of dicts): {str(col_value)[:100]}")

            elif col_name in ['state', 'actions']:
                print(f"Data type: {type(col_value)}")
                print(f"NumPy dtype: {col_value.dtype}")
                print(f"Value: {col_value}")
            else: # For timestamp, frame_index, episode_index, index, task_index
                # Avoid printing binary data directly
                if isinstance(col_value, bytes):
                    print(f"Value: <binary data of {len(col_value)} bytes>")
                elif isinstance(col_value, dict) and 'bytes' in col_value:
                    print(f"Value: <dictionary containing binary data, {len(col_value['bytes'])} bytes>")
                    if 'path' in col_value:
                        print(f"Path: {col_value['path']}")
                else:
                    print(f"Value: {col_value}")

    print("\n=== Columns and Data Types ===")
    print(df.dtypes) # 更简洁地只看列名和数据类型

    # print("\n=== Describe (统计信息) ===") # You can uncomment this if needed
    # print(df.describe())

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# 方法二：使用 pyarrow 直接查看 Parquet 文件的 schema (更轻量级，无需加载全部数据)
# try:
#     parquet_file = pq.ParquetFile(file_path)
#     print("\n=== Parquet Schema (via pyarrow) ===")
#     print(parquet_file.schema) # 查看 Parquet 文件的详细 schema
#     print("\n=== Parquet Metadata (via pyarrow) ===")
#     print(parquet_file.metadata) # 查看 Parquet 文件的元数据，包括列、行组等信息
# except FileNotFoundError:
#     print(f"Error: File not found at {file_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")