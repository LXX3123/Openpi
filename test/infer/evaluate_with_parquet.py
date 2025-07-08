import json
import logging
import sys
import os
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from openpi_client import websocket_client_policy as _websocket_client_policy

sys.path.append('/home/adminroot/lxx/openpi/code/openpi/test/parquet_2_json/uav_flow')
from parquet_2_json import process_parquet_episode
# 添加路径以导入视频生成脚本
sys.path.append('/home/adminroot/lxx/openpi/code/openpi/test/infer/trajectory_video/py')
from trajectory_video_new import generate_video_from_trajectory

def get_prompt_from_task_index(tasks_jsonl_path, task_index):
    """
    从 tasks.jsonl 文件中根据 task_index 查找并返回对应的 prompt。
    """
    try:
        with open(tasks_jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('task_index') == task_index:
                    return data.get('task')
    except FileNotFoundError:
        logging.error(f"tasks.jsonl 文件未找到: {tasks_jsonl_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"解析 tasks.jsonl 文件时出错: {tasks_jsonl_path}")
        return None
    
    logging.warning(f"在 {tasks_jsonl_path} 中未找到 task_index {task_index} 对应的任务。")
    return None

def calculate_trajectory_deviation(traj1, traj2):
    """
    计算两条轨迹的位置和姿态偏差。
    轨迹应为 (N, 6) 的形状，其中 N 是点数，6是 (x, y, z, roll, pitch, yaw)。
    """
    pos1, rot1 = traj1[:, :3], traj1[:, 3:]
    pos2, rot2 = traj2[:, :3], traj2[:, 3:]

    positional_errors = np.linalg.norm(pos1 - pos2, axis=1)
    positional_mae = np.mean(positional_errors)

    rotational_errors = np.abs(rot1 - rot2)
    rotational_mae = np.mean(rotational_errors)

    return positional_mae, rotational_mae, positional_errors


def visualize_trajectories_3d(traj1, traj2):
    """
    使用 matplotlib 可视化两条3D轨迹。
    """
    pos1 = traj1[:, :3]
    pos2 = traj2[:, :3]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 交换X和Y轴以匹配俯视图
    ax.plot(pos1[:, 1], pos1[:, 0], pos1[:, 2], 'o-', label='Original Trajectory', color='blue')
    ax.plot(pos2[:, 1], pos2[:, 0], pos2[:, 2], 'o-', label='Inferred Trajectory', color='red')

    for i in range(len(pos1)):
        ax.plot([pos1[i, 1], pos2[i, 1]], [pos1[i, 0], pos2[i, 0]], [pos1[i, 2], pos2[i, 2]],
                '--', color='gray', linewidth=0.8)

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    plt.show()

def visualize_trajectories_2d_topdown(traj1, traj2):
    """
    使用 matplotlib 可视化两条轨迹的俯视图。
    横轴为Y (左/右), 纵轴为X (前/后)
    """
    pos1 = traj1[:, :2]  # 只取 X, Y 坐标
    pos2 = traj2[:, :2]  # 只取 X, Y 坐标

    fig, ax = plt.subplots(figsize=(10, 10))

    # 交换X和Y轴进行绘图
    ax.plot(pos1[:, 1], pos1[:, 0], 'o-', label='Original Trajectory', color='blue')
    ax.plot(pos2[:, 1], pos2[:, 0], 'o-', label='Inferred Trajectory', color='red')

    for i in range(len(pos1)):
        ax.plot([pos1[i, 1], pos2[i, 1]], [pos1[i, 0], pos2[i, 0]],
                '--', color='gray', linewidth=0.8)

    ax.set_xlabel('Y (m, Left/Right)')
    ax.set_ylabel('X (m, Forward/Backward)')
    ax.set_title('Trajectory Comparison (Top-down View)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def main():
    HOST = "0.0.0.0"
    PORT = 8000
    # 将此路径更改为指向 Parquet 文件
    # EPISODE_DATA_PATH = "/home/adminroot/lxx/dataset/uav_flow_lerobot_format/fixed_command/pass/train_without_first_10/uav_flow/data/chunk-000/episode_000022.parquet"
    # EPISODE_DATA_PATH = "/home/adminroot/lxx/dataset/uav_flow_lerobot_format/train/uav_flow/data/chunk-000/episode_000020.parquet"
    # 这条数据感觉有问题
    # EPISODE_DATA_PATH = "/home/adminroot/lxx/dataset/uav_flow_lerobot_format/fixed_command/train0621/uav_flow/data/chunk-000/episode_000102.parquet"
    EPISODE_DATA_PATH = "/home/adminroot/lxx/dataset/uav_flow_lerobot_format/fixed_command/pass/test/uav_flow/data/chunk-000/episode_000027.parquet"
    REPLAN_STEPS = 10  # 每次推理后取前10个动作
    IGNORE_FIRST_10 = True  # 开关：True则忽略前10条数据，False则保持原样

    logging.info(f"正在从 {EPISODE_DATA_PATH} 加载并处理 Parquet 数据...")
    
    # 定义提取出的图像的存放位置。评估脚本需要这些位于磁盘上的图像。
    output_dir_for_images = "/home/adminroot/lxx/openpi/code/openpi/test/infer/infer_output"
    
    # 调用函数来处理 Parquet 文件。
    # 它会将图像保存到 `output_dir_for_images` 并返回所需的数据结构。
    episode_data = process_parquet_episode(EPISODE_DATA_PATH, output_dir_for_images, save_json=True)

    if IGNORE_FIRST_10:
        if len(episode_data) > 10:
            logging.info("开关为True，正在忽略原始数据中的前10条数据...")
            episode_data = episode_data[10:]
        else:
            logging.warning(f"数据总数不足10条 (只有 {len(episode_data)} 条)，无法忽略。将使用所有数据。")

    if not episode_data:
        logging.error("未能从 Parquet 文件加载数据。正在中止。")
        return

    # 1. 从 episode 数据中获取 task_index
    if 'task_index' not in episode_data[0] or episode_data[0]['task_index'] is None:
        logging.error("在 episode 数据的第一帧中未找到 'task_index'。")
        return
    task_index = episode_data[0]['task_index']

    # 2. 构造 tasks.jsonl 的路径
    try:
        # 假设路径结构为 .../test/uav_flow/data/chunk-xxx/episode_xxxxx.parquet
        # 我们需要 .../test/uav_flow/meta/tasks.jsonl
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(EPISODE_DATA_PATH)))
        tasks_jsonl_path = os.path.join(base_path, 'meta', 'tasks.jsonl')
    except Exception as e:
        logging.error(f"无法从 EPISODE_DATA_PATH 推断 'meta' 目录路径: {EPISODE_DATA_PATH}. Error: {e}")
        return

    # 3. 获取 PROMPT
    PROMPT = get_prompt_from_task_index(tasks_jsonl_path, task_index)
    if PROMPT is None:
        logging.error("无法获取 PROMPT。正在中止。")
        return
    
    logging.info(f"成功获取任务提示: '{PROMPT}'")

    MAX_STEPS = len(episode_data)  # 将总步数设置为数据中的对象数量

    # 构造原始轨迹
    # 原始轨迹由第0帧的状态和之后每一帧的动作组成
    # 注意：在数据集中，frame_i 的 state 等于 frame_i-1 的 actions
    # 因此，真实的轨迹点是所有 object 的 state 加上最后一个 object 的 actions
    original_trajectory_states = [np.array(d['state']) for d in episode_data]
    original_trajectory_states.append(np.array(episode_data[-1]['actions']))
    original_trajectory = np.array(original_trajectory_states)


    logging.info(f"正在连接到服务器 ws://{HOST}:{PORT}")
    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    all_inferred_actions = []
    
    # 首次推理
    # 使用第一个 object 的数据进行初始化
    current_state = np.array(episode_data[0]['state'])
    image_path = episode_data[0]['image_path']
    
    logging.info("开始循环推理...")
    for i in range(0, MAX_STEPS, REPLAN_STEPS):
        logging.info(f"正在执行推理步: {i+1}-{i+REPLAN_STEPS}")
        
        # 构造输入 example
        # image 固定不变, wrist_image 和 state 变化
        wrist_image_path = episode_data[i]['wrist_image_path']
        
        element = {
            "observation/image": imageio.imread(image_path),
            "observation/wrist_image": imageio.imread(wrist_image_path),
            "observation/state": current_state,
            "prompt": PROMPT
        }

        # 调用 server 进行推理
        action_chunk = client.infer(element)["actions"]
        all_inferred_actions.extend(action_chunk)

        # 更新下一次推理的状态
        # 使用返回的动作序列的最后一个点的6D位姿作为下一次的起始状态
        current_state = np.array(action_chunk[-1][:6])

    logging.info("推理完成。")

    # 构造推理轨迹
    # 推理出的轨迹是初始状态加上所有推理出的动作
    inferred_trajectory_states = [np.array(episode_data[0]['state'])]
    inferred_trajectory_states.extend([np.array(act[:6]) for act in all_inferred_actions])
    inferred_trajectory = np.array(inferred_trajectory_states)
    
    # 确保两条轨迹长度一致以便比较
    min_len = min(len(original_trajectory), len(inferred_trajectory))
    original_trajectory = original_trajectory[:min_len]
    inferred_trajectory = inferred_trajectory[:min_len]

    # 计算偏差并输出结果
    pos_mae, rot_mae, pos_errors = calculate_trajectory_deviation(original_trajectory, inferred_trajectory)

    print("\n--- 轨迹偏差分析 ---")
    print(f"平均绝对位置误差 (Positional MAE): {pos_mae:.4f}")
    print(f"平均绝对姿态误差 (Rotational MAE): {rot_mae:.4f}")
    print("\n每个点的具体位置误差:")
    for i, err in enumerate(pos_errors):
        print(f"  点 {i}: {err:.4f}")
    print("------------------------\n")

    # 可视化轨迹
    logging.info("正在生成轨迹对比图...")
    visualize_trajectories_3d(original_trajectory, inferred_trajectory)
    visualize_trajectories_2d_topdown(original_trajectory, inferred_trajectory)    

    # 从EPISODE_DATA_PATH动态生成输出JSON文件名
    episode_match = re.search(r'episode_(\d+)\.parquet$', EPISODE_DATA_PATH)
    if episode_match:
        episode_num = episode_match.group(1)
        output_filename = f"inferred_trajectory_{episode_num}.json"
    else:
        output_filename = "inferred_trajectory.json" # Fallback

    output_path = f"/home/adminroot/lxx/openpi/code/openpi/test/infer/trajectory_video/{output_filename}"
    logging.info(f"正在保存推理轨迹到 {output_path}...")

    # 准备用于JSON输出的数据列表
    output_data = []
    # 遍历原始的 episode 数据，以其长度为准
    for i in range(len(episode_data)):
        # 确保我们不会因为推理出的点数较少而出错
        if i < len(inferred_trajectory):
            # inferred_trajectory 的第一个点是初始状态，后续的点才是动作结果。
            # 因此，第 i 个 object 对应的 action 是 inferred_trajectory 的第 i+1 个点。
            # 我们需要检查 i+1 是否仍在 inferred_trajectory 的范围内。
            if (i + 1) < len(inferred_trajectory):
                action = inferred_trajectory[i+1].tolist()
            else:
                # 如果没有对应的推理动作（例如在最后一个点），则设为 None 或空列表
                action = None
        else:
            action = None

        output_data.append({
            "prompt": PROMPT,
            "wrist_image_path": episode_data[i]['wrist_image_path'],
            "state": episode_data[i]['state'],
            "actions": action
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logging.info(f"推理轨迹已保存到 {output_path}")

    # 自动调用脚本生成视频
    logging.info("正在调用脚本生成轨迹视频...")
    try:
        generate_video_from_trajectory(output_path)
        logging.info("视频生成完毕。")
    except Exception as e:
        logging.error(f"生成视频时出错: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()