import pandas as pd
from PIL import Image
import io
import json
import os

def process_parquet_episode(file_path, output_dir, save_json=True):
    """
    处理单个 Parquet episode 文件。它会提取图像并保存，然后返回 episode 数据。

    Args:
        file_path (str): 输入的 Parquet 文件路径。
        output_dir (str): 用于保存图像和 JSON 文件的输出目录。
        save_json (bool): 如果为 True，则将 episode 数据保存为 JSON 文件。

    Returns:
        list: 一个包含字典的列表，每个字典代表一帧的数据。
    """
    # 如果输出目录不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)

    # 加载 Parquet 文件
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"无法加载 Parquet 文件 {file_path}: {e}")
        return []

    print(f"正在处理 Parquet 文件中的 {len(df)} 行: {file_path}")

    all_rows_data = []
    output_episode_index_for_filename = None
    image_output_dir = None # 将在第一个有效行上设置

    for index, row in df.iterrows():
        try:
            current_episode_idx = row['episode_index']
            current_frame_idx = row['frame_index']
        except KeyError as e:
            print(f"在第 {index} 行缺少预期的列 {e}。请确保 Parquet 文件包含 'episode_index' 和 'frame_index'。正在跳过此行。")
            continue

        # 在第一个有效行上，创建特定于 episode 的图像目录
        if image_output_dir is None:
            image_output_dir = os.path.join(output_dir, f'images_ep{current_episode_idx}')
            os.makedirs(image_output_dir, exist_ok=True)

        if output_episode_index_for_filename is None:
            output_episode_index_for_filename = current_episode_idx

        # print(f"正在处理 episode_index: {current_episode_idx}, frame_index: {current_frame_idx}")

        image_data = row.get('image')
        wrist_image_data = row.get('wrist_image')
        task_idx = row.get('task_index')
        
        image_filepath = None
        wrist_image_filepath = None

        # --- 处理主图像 ---
        image_bytes = image_data.get('bytes') if isinstance(image_data, dict) else image_data
        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image_filename = f"image_ep{current_episode_idx}_frame_{current_frame_idx}.png"
                image_filepath = os.path.join(image_output_dir, image_filename)
                image.save(image_filepath)
            except Exception as e:
                print(f"无法为 episode {current_episode_idx}, frame {current_frame_idx} 打开或保存图像: {e}")
                image_filepath = None
        
        # --- 处理手腕图像 ---
        wrist_image_bytes = wrist_image_data.get('bytes') if isinstance(wrist_image_data, dict) else wrist_image_data
        if wrist_image_bytes:
            try:
                wrist_image = Image.open(io.BytesIO(wrist_image_bytes))
                wrist_image_filename = f"wrist_image_ep{current_episode_idx}_frame_{current_frame_idx}.png"
                wrist_image_filepath = os.path.join(image_output_dir, wrist_image_filename)
                wrist_image.save(wrist_image_filepath)
            except Exception as e:
                print(f"无法为 episode {current_episode_idx}, frame {current_frame_idx} 打开或保存手腕图像: {e}")
                wrist_image_filepath = None

        current_row_output_data = {
            "episode_index": int(current_episode_idx),
            "frame_index": int(current_frame_idx),
            "image_path": os.path.abspath(image_filepath) if image_filepath else None,
            "wrist_image_path": os.path.abspath(wrist_image_filepath) if wrist_image_filepath else None,
            "task_index": int(task_idx) if task_idx is not None else None,
            "state": row['state'].tolist() if 'state' in row and hasattr(row['state'], 'tolist') else row.get('state'),
            "actions": row['actions'].tolist() if 'actions' in row and hasattr(row['actions'], 'tolist') else row.get('actions'),
            "timestamp": float(row['timestamp']) if 'timestamp' in row else None,
        }
        all_rows_data.append(current_row_output_data)

    if save_json and all_rows_data:
        if output_episode_index_for_filename is not None:
            data_output_filename = f"episode_data_{output_episode_index_for_filename}.json"
        else:
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            data_output_filename = f"{file_basename}_data.json"
            
        data_output_filepath = os.path.join(output_dir, data_output_filename)
        try:
            with open(data_output_filepath, 'w') as f:
                json.dump(all_rows_data, f, indent=2)
            print(f"\n已将场景的采集数据保存至: {data_output_filepath}")
        except Exception as e:
            print(f"无法将采集的数据保存到 {data_output_filepath}: {e}")
    elif not all_rows_data:
        print("没有处理任何数据以供保存。")

    return all_rows_data

if __name__ == "__main__":
    # 当作为主脚本运行时，保持原有行为
    # 定义输出目录
    output_dir = '/home/adminroot/lxx/openpi/code/openpi/test/parquet_2_json/uav_flow/output'
    # 定义 Parquet 文件路径
    file_path = '/home/adminroot/lxx/dataset/uav_flow_lerobot_format/fixed_command/pass/train/uav_flow/data/chunk-000/episode_000001.parquet'
    
    # 调用函数并保存 JSON 文件
    process_parquet_episode(file_path, output_dir, save_json=True)