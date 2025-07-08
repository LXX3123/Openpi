import json
import logging

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from openpi_client import websocket_client_policy as _websocket_client_policy


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


def visualize_trajectories(traj1, traj2):
    """
    使用 matplotlib 可视化两条3D轨迹。
    """
    pos1 = traj1[:, :3]
    pos2 = traj2[:, :3]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], 'o-', label='Original Trajectory', color='blue')
    ax.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], 'o-', label='Inferred Trajectory', color='red')

    for i in range(len(pos1)):
        ax.plot([pos1[i, 0], pos2[i, 0]], [pos1[i, 1], pos2[i, 1]], [pos1[i, 2], pos2[i, 2]],
                '--', color='gray', linewidth=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    plt.show()


def main():
    HOST = "0.0.0.0"
    PORT = 8000
    EPISODE_DATA_PATH = "/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/uav_flow/episode_data_0.json"
    PROMPT = "Please proceed toward the front of the tree"
    REPLAN_STEPS = 10  # 每次推理后取前10个动作

    logging.info(f"正在从 {EPISODE_DATA_PATH} 加载数据...")
    with open(EPISODE_DATA_PATH, 'r') as f:
        episode_data = json.load(f)

    MAX_STEPS = len(episode_data)  # 将总步数设置为数据中的对象数量

    # 1. 构造原始轨迹
    # 原始轨迹由第0帧的状态和之后每一帧的动作组成
    # 注意：在数据集中，frame_i 的 state 等于 frame_i-1 的 actions
    # 因此，真实的轨迹点是所有 object 的 state 加上最后一个 object 的 actions
    original_trajectory_states = [np.array(d['state']) for d in episode_data]
    original_trajectory_states.append(np.array(episode_data[-1]['actions']))
    original_trajectory = np.array(original_trajectory_states)


    logging.info(f"正在连接到服务器 ws://{HOST}:{PORT}")
    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    all_inferred_actions = []
    
    # 2. 首次推理
    # 使用第一个 object 的数据进行初始化
    current_state = np.array(episode_data[0]['state'])
    image_path = episode_data[0]['image_path']
    wrist_image_path = episode_data[0]['wrist_image_path']
    
    logging.info("开始循环推理...")
    for i in range(0, MAX_STEPS, REPLAN_STEPS):
        logging.info(f"正在执行推理步: {i+1}-{i+REPLAN_STEPS}")
        
        # 3. 构造输入 example
        # image 固定不变, wrist_image 和 state 变化
        wrist_image_path = episode_data[i]['wrist_image_path']
        
        element = {
            "observation/image": imageio.imread(image_path),
            "observation/wrist_image": imageio.imread(wrist_image_path),
            "observation/state": current_state,
            "prompt": PROMPT
        }

        # 4. 调用 server 进行推理
        action_chunk = client.infer(element)["actions"]
        all_inferred_actions.extend(action_chunk)

        # 5. 更新下一次推理的状态
        # 使用返回的动作序列的最后一个点的6D位姿作为下一次的起始状态
        current_state = np.array(action_chunk[-1][:6])

    logging.info("推理完成。")

    # 6. 构造推理轨迹
    # 推理出的轨迹是初始状态加上所有推理出的动作
    inferred_trajectory_states = [np.array(episode_data[0]['state'])]
    inferred_trajectory_states.extend([np.array(act[:6]) for act in all_inferred_actions])
    inferred_trajectory = np.array(inferred_trajectory_states)
    
    # 确保两条轨迹长度一致以便比较
    min_len = min(len(original_trajectory), len(inferred_trajectory))
    original_trajectory = original_trajectory[:min_len]
    inferred_trajectory = inferred_trajectory[:min_len]

    # 7. 计算偏差并输出结果
    pos_mae, rot_mae, pos_errors = calculate_trajectory_deviation(original_trajectory, inferred_trajectory)

    print("\n--- 轨迹偏差分析 ---")
    print(f"平均绝对位置误差 (Positional MAE): {pos_mae:.4f}")
    print(f"平均绝对姿态误差 (Rotational MAE): {rot_mae:.4f}")
    print("\n每个点的具体位置误差:")
    for i, err in enumerate(pos_errors):
        print(f"  点 {i}: {err:.4f}")
    print("------------------------\n")

    # 8. 可视化轨迹
    logging.info("正在生成轨迹对比图...")
    visualize_trajectories(original_trajectory, inferred_trajectory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()