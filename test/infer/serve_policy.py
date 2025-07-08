import logging
import socket

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def main() -> None:
    """
    加载基于UAV微调的策略模型，并启动一个WebSocket服务器。
    """
    logging.info("正在加载策略模型...")

    # 从 infer_uav.py 中获取模型配置和检查点路径
    config = _config.get_config("pi0_uav_low_mem_finetune")
    checkpoint_dir = "/home/adminroot/lxx/openpi/code/openpi/checkpoints/pi0_uav_low_mem_finetune/my_uav_flow_lora_experiment/59999"

    # 创建一个训练好的策略
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
        logging.warning(f"无法解析主机名 {hostname}, 使用 {local_ip} 代替。")

    logging.info("正在创建服务器 (host: %s, ip: %s)", hostname, local_ip)

    # 启动 WebSocket 服务器
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
        metadata=policy_metadata,
    )
    logging.info("服务器已在 ws://%s:8000 上启动", local_ip)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()