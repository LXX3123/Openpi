import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
from PIL import Image
import os # Added for directory creation

# 假设你的 Parquet 文件名为 'your_dataset.parquet'
file_path = '/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/libero/episode_000000.parquet'

# 方法一：使用 pandas 读取文件并查看信息
try:
    df = pd.read_parquet(file_path)

    print("=== DataFrame Info ===")
    df.info()  # 查看列名、数据类型、非空值数量

    # Create output directory if it doesn't exist
    output_image_dir = "/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/libero/images"
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
                            image = Image.open(BytesIO(image_bytes))
                            print(f"{col_name.capitalize()} format: {image.format}, size: {image.size}, mode: {image.mode}")
                            image_filename = os.path.join(output_image_dir, f"row_{i}_{col_name}.png")
                            image.save(image_filename)
                            print(f"{col_name.capitalize()} saved to {image_filename}")
                        else:
                            print(f"{col_name.capitalize()} data: No image bytes found in dictionary.")
                    elif isinstance(col_value, bytes): # Handle if it's directly bytes (less common for Parquet dicts)
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
                        print(f"{col_name.capitalize()} data: {str(col_value)[:200]} (Type: {type(col_value)})") # Print limited string representation
                except Exception as e:
                    print(f"Error processing {col_name} for row {i}: {e}. Value was: {str(col_value)[:100]}")
            elif col_name in ['state', 'actions']:
                print(f"Data type: {type(col_value)}")
                print(f"Value: {col_value}")
            else: # For timestamp, frame_index, episode_index, index, task_index
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