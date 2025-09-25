import os
import pathlib

import jax
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX devices:", jax.devices())

# 你的模型路径
checkpoint_dir = "/data/home/zhangjing2/th/openpi/pytorch_pi05_leboro"

# 检查模型文件是否存在
model_path = pathlib.Path(checkpoint_dir) / "model.safetensors"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading model from: {checkpoint_dir}")

# 使用 LIBERO 配置（根据你的路径推测）
config = _config.get_config("pi05_libero")

# 创建训练好的策略
print("Creating policy...")
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

print("Policy created successfully!")

# 创建一个虚拟示例进行测试
print("Creating test example...")
# 对于 LIBERO，我们需要创建一个合适的示例
# 这里我们创建一个基本的示例结构
import numpy as np

example = {
    "observation": {
        "image": {
            "cam_high": np.ones((224, 224, 3), dtype=np.uint8),
            "cam_low": np.ones((224, 224, 3), dtype=np.uint8),
        },
        "state": np.ones((7,), dtype=np.float32),  # 7维状态向量
    },
    "prompt": "pick up the red block",
}

# 运行推理
print("Running inference...")
result = policy.infer(example)

# 输出结果
print("Inference completed!")
print("Actions shape:", result["actions"].shape)
print("Actions:", result["actions"])

# 清理内存
del policy
print("Policy deleted, memory freed.")