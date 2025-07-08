from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import imageio.v2 as imageio

config = config.get_config("pi0_libero_low_mem_finetune")
checkpoint_dir = "/home/adminroot/lxx/openpi/code/openpi/checkpoints/pi0_libero_low_mem_finetune/my_libero_lora_experiment_spatial_new/20000"

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/image": imageio.imread("/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/libero/images/row_0_image.png"),
    "observation/wrist_image": imageio.imread("/home/adminroot/lxx/openpi/code/openpi/test/dataset_transfer/libero/images/row_0_wrist_image.png"),
    "observation/state": np.array([
      -0.20487476885318756,
      -0.010081525892019272,
      1.174657940864563,
      3.1396350860595703,
      0.0001441165222786367,
      -0.08801795542240143,
      0.03878617286682129,
      -0.03878973424434662
    ]),
    "prompt": "pick up the black bowl next to the cookie box and place it on the plate"
}
action_chunk = policy.infer(example)["actions"]
with open("/home/adminroot/lxx/openpi/code/openpi/test/action_chunk.txt", "w") as f:
    f.write(str(action_chunk))