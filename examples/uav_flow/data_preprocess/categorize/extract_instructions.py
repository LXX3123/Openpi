"""
该脚本用于从一系列 Parquet 文件中提取所有唯一的 'instruction_unified' 字段，
并将每个 'id' 与其对应的 'instruction_unified' 进行配对。

它会读取指定目录中所有匹配 'train-*.parquet' 模式的 Parquet 文件，
然后执行两个任务：
1. 将所有唯一的 'instruction_unified' 值写入到 'instruction_unified.txt'，每个值占一行。
2. 将每一行中的 'id' 和 'instruction_unified' 配对，用分号分隔，写入到 'id_and_instruction_unified.txt'，不进行去重。

用法:
uv run examples/uav_flow/extract_instructions.py --data_dir /path/to/your/uav_parquet_files_folder

请确保已安装 pandas 和 pyarrow:
`uv pip install pandas pyarrow`
"""

import os
import glob
import pandas as pd
import json
import tyro
import logging
from tqdm import tqdm
from tqdm.auto import tqdm

# 为 pandas 的 apply 方法启用tqdm进度条
tqdm.pandas()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(data_dir: str):
    """
    主函数，执行指令提取和写入操作。

    Args:
        data_dir (str): 包含 Parquet 文件的目录路径。
    """
    unique_instruction_output_file = "/home/adminroot/lxx/openpi/code/openpi/examples/uav_flow/instruction_unified.txt"
    id_instruction_output_file = "/home/adminroot/lxx/openpi/code/openpi/examples/uav_flow/id_and_instruction_unified.txt"
    
    # 查找所有 Parquet 文件
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    if not parquet_files:
        logging.error(f"在目录 '{data_dir}' 中未找到匹配 'train-*.parquet' 的文件。")
        return

    logging.info(f"找到 {len(parquet_files)} 个 Parquet 文件。")

    unique_instructions = set()
    total_pairs_written = 0
    seen_id_instruction_pairs = set() # 用于跟踪已写入的 (id, instruction) 对

    # 确保输出目录存在
    output_dir = os.path.dirname(unique_instruction_output_file)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 打开第二个输出文件，准备写入 id 和 instruction 对
        with open(id_instruction_output_file, 'w', encoding='utf-8') as id_f:
            # 使用 tqdm 显示处理进度
            for pq_file in tqdm(parquet_files, desc="处理 Parquet 文件"):
                try:
                    # 读取 id 和 log 两列
                    df = pd.read_parquet(pq_file, columns=['id', 'log'])
                    
                    # 定义一个函数来安全地解析 JSON 和提取指令
                    def get_instruction(log_str):
                        try:
                            return json.loads(log_str).get('instruction_unified')
                        except (json.JSONDecodeError, AttributeError):
                            return None

                    # 应用函数创建新列，并显示进度条
                    df['instruction'] = df['log'].progress_apply(get_instruction)
                    
                    # --- 任务1: 更新唯一指令集合 ---
                    # 过滤掉空指令后，获取唯一值并更新集合
                    valid_instructions = df['instruction'].dropna().unique()
                    unique_instructions.update(valid_instructions)

                    # --- 任务2: 写入 id 和指令对 (去重) ---
                    # 过滤掉没有成功提取到指令的行
                    pairs_df = df[['id', 'instruction']].dropna()
                    # 首先在当前文件中去重，减少循环次数
                    pairs_df.drop_duplicates(inplace=True)

                    # 遍历去重后的配对，并检查是否已在全局集合中
                    for _, row in tqdm(pairs_df.iterrows(), total=pairs_df.shape[0], desc=f"去重写入 {os.path.basename(pq_file)}", leave=False):
                        pair = (row['id'], row['instruction'])
                        if pair not in seen_id_instruction_pairs:
                            id_f.write(f"{pair[0]};{pair[1]}\n")
                            seen_id_instruction_pairs.add(pair)
                            total_pairs_written += 1

                except Exception as e:
                    logging.error(f"处理文件 {pq_file} 时出错: {e}")
                    continue
    except IOError as e:
        logging.error(f"无法写入文件 {id_instruction_output_file}: {e}")
        return
    
    logging.info(f"成功将 {total_pairs_written} 条 id-指令配对写入到 {id_instruction_output_file}")

    # --- 写入唯一的指令文件 ---
    if not unique_instructions:
        logging.warning("未能提取到任何唯一指令。")
    else:
        try:
            with open(unique_instruction_output_file, 'w', encoding='utf-8') as f:
                for instruction in sorted(list(unique_instructions)): # 排序以保证输出顺序一致
                    f.write(instruction + '\n')
            logging.info(f"成功将 {len(unique_instructions)} 条唯一指令写入到 {unique_instruction_output_file}")
        except IOError as e:
            logging.error(f"无法写入文件 {unique_instruction_output_file}: {e}")


if __name__ == "__main__":
    tyro.cli(main)