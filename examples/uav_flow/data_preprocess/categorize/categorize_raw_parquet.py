import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import glob
import json
from collections import defaultdict
from tqdm import tqdm # 导入tqdm

# --- 配置 ---
INPUT_DIR = "/home/adminroot/lxx/dataset/uav_flow/raw"
OUTPUT_DIR_BASE = "/home/adminroot/lxx/dataset/uav_flow/categorized"
MAPPING_FILE = "/home/adminroot/lxx/openpi/code/openpi/examples/uav_flow/instruction_unified_and_catetory_final.txt"
GROUPS_PER_FILE = 100
# 建议根据您的内存大小和数据特性调整批处理大小
BATCH_SIZE = 10000 

# 类别名称到目录名称的映射
CATEGORY_TO_DIR = {
    "Move": "move",
    "Turn": "turn",
    "Land": "land",
    "Pass": "pass",
    "Retreat": "retreat",
    "Approach": "approach",
    "Surround": "surround",
    "Ascend/Descend": "ascend_descend",
    "Rotate": "rotate",
    "Shift": "shift"
}

def load_instruction_mapping(file_path):
    """加载指令到类别的映射"""
    mapping = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ';' in line:
                    instruction, category = line.rsplit(';', 1)
                    mapping[instruction.strip()] = category.strip()
    except FileNotFoundError:
        print(f"错误: 映射文件未找到: {file_path}")
        exit(1)
    return mapping

def ensure_output_dirs():
    """确保所有输出目录都存在"""
    for dir_name in CATEGORY_TO_DIR.values():
        path = os.path.join(OUTPUT_DIR_BASE, dir_name)
        os.makedirs(path, exist_ok=True)

def process_group(group_df, writers, instruction_map):
    """处理单个数据组：分类并写入文件"""
    if group_df.empty:
        return

    # 获取指令
    try:
        # group_df['log'] 在这里已经是字符串了
        first_row_log_str = group_df.iloc[0]['log']
        first_row_log = json.loads(first_row_log_str)
        instruction = first_row_log.get("instruction_unified")
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        group_id = group_df.iloc[0]['id'] if not group_df.empty else "N/A"
        print(f"警告: 解析指令失败，跳过组 ID: {group_id}。错误: {e}")
        return

    if not instruction or instruction not in instruction_map:
        group_id = group_df.iloc[0]['id']
        print(f"警告: 找不到指令 '{instruction}' 的类别，跳过组 ID: {group_id}")
        return

    # 找到类别和对应的目录
    category = instruction_map[instruction]
    category_dir_name = CATEGORY_TO_DIR.get(category)

    if not category_dir_name:
        group_id = group_df.iloc[0]['id']
        print(f"警告: 找不到指令类别 '{category}' 的目录，跳过组 ID: {group_id}")
        return

    # 获取或创建此类的写入器
    if category_dir_name not in writers:
        writers[category_dir_name] = {'writer': None, 'group_count': 0, 'file_index': 0, 'schema': None}

    writer_info = writers[category_dir_name]

    # 检查是否需要创建新文件
    if writer_info['writer'] is None or writer_info['group_count'] >= GROUPS_PER_FILE:
        if writer_info['writer'] is not None:
            writer_info['writer'].close()
            writer_info['file_index'] += 1
        
        output_filename = f"{category_dir_name}-{writer_info['file_index']:05d}.parquet"
        output_path = os.path.join(OUTPUT_DIR_BASE, category_dir_name, output_filename)
        
        # 从当前数据帧推断schema
        arrow_schema = pa.Table.from_pandas(group_df).schema
        writer_info['schema'] = arrow_schema
        
        writer_info['writer'] = pq.ParquetWriter(output_path, writer_info['schema'])
        writer_info['group_count'] = 0
        print(f"\n为类别 '{category_dir_name}' 创建新文件: {output_path}")

    # 将当前组写入文件
    arrow_table = pa.Table.from_pandas(group_df, schema=writer_info['schema'], preserve_index=False)
    writer_info['writer'].write_table(arrow_table)
    writer_info['group_count'] += 1


def main():
    """主处理函数"""
    print("开始处理...")
    instruction_map = load_instruction_mapping(MAPPING_FILE)
    print(f"成功加载 {len(instruction_map)} 条指令映射。")
    ensure_output_dirs()
    print(f"已确保所有输出目录在 '{OUTPUT_DIR_BASE}' 中创建。")

    writers = {}
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not input_files:
        print(f"警告: 在 '{INPUT_DIR}' 中没有找到 .parquet 文件。")
        return
    print(f"找到 {len(input_files)} 个输入 Parquet 文件。")

    # 使用tqdm包装文件列表以显示总体进度
    for file_path in tqdm(input_files, desc="文件处理进度"):
        try:
            parquet_file = pq.ParquetFile(file_path)
            
            current_group_id = None
            # group_rows现在将存储字典，以减少内存开销
            group_rows = []

            # 计算总批次数以显示文件内部进度
            num_batches = (parquet_file.metadata.num_rows + BATCH_SIZE - 1) // BATCH_SIZE
            
            # 迭代所有批次，不指定列以加载所有数据
            batch_iterator = parquet_file.iter_batches(batch_size=BATCH_SIZE)
            
            # 使用tqdm包装批次迭代器，显示单个文件的处理进度
            pbar = tqdm(batch_iterator, total=num_batches, desc=f"处理中 {os.path.basename(file_path)}", leave=False)
            for batch in pbar:
                # 将批次转换为记录列表（字典列表），比DataFrame更轻量
                records = batch.to_pylist()
                for row in records:
                    row_id = row['id']
                    if current_group_id is not None and row_id != current_group_id:
                        # 新ID出现，处理已收集的组
                        group_df = pd.DataFrame(group_rows)
                        process_group(group_df, writers, instruction_map)
                        group_rows = []

                    current_group_id = row_id
                    group_rows.append(row)
            
            # 处理文件末尾的最后一组
            if group_rows:
                group_df = pd.DataFrame(group_rows)
                process_group(group_df, writers, instruction_map)

        except Exception as e:
            print(f"\n处理文件时发生严重错误 {file_path}: {e}")
            continue

    # 关闭所有打开的文件写入器
    for writer_info in writers.values():
        if writer_info['writer']:
            writer_info['writer'].close()

    print("\n处理完成。")

if __name__ == "__main__":
    main()