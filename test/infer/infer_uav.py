from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import imageio.v2 as imageio

config = config.get_config("pi0_uav_low_mem_finetune")
checkpoint_dir = "/home/adminroot/lxx/openpi/code/openpi/checkpoints/pi0_uav_low_mem_finetune/my_uav_flow_lora_experiment/59999"
# config = config.get_config("pi0_libero_low_mem_finetune")
# checkpoint_dir = "/home/adminroot/lxx/openpi/code/openpi/checkpoints/pi0_libero_low_mem_finetune/my_libero_lora_experiment_spatial_new/25000"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/image": imageio.imread("/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/uav_flow/images/image_ep0_frame_0.png"),
    "observation/wrist_image": imageio.imread("/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/uav_flow/images/wrist_image_ep0_frame_0.png"),
    "observation/state": np.array([
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]),
    "prompt": "Please proceed toward the front of the tree"
}
action_chunk = policy.infer(example)["actions"]
with open("/home/adminroot/lxx/openpi/code/openpi/test/action_chunk.txt", "w") as f:
    f.write(str(action_chunk))