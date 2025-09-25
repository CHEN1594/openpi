import os
import pathlib
import numpy as np

import jax
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.shared import download

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

# 使用 LIBERO 配置
config = _config.get_config("pi05_libero")

# 直接创建模型，绕过 policy 的归一化
print("Loading model directly...")
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
model = config.model.load_pytorch(config, weight_path)
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

print("Model loaded successfully!")

# 创建一个虚拟示例进行测试
print("Creating test example...")
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

# 手动处理数据预处理
import torch
from openpi.models.model import Observation

# 转换数据格式
images = {
    "cam_high": torch.from_numpy(example["observation"]["image"]["cam_high"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
    "cam_low": torch.from_numpy(example["observation"]["image"]["cam_low"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
}

# 创建图像掩码（假设所有像素都有效）
image_masks = {
    "cam_high": torch.ones(1, dtype=torch.bool),
    "cam_low": torch.ones(1, dtype=torch.bool),
}

# 创建状态
state = torch.from_numpy(example["observation"]["state"]).unsqueeze(0).float()

# 创建观察对象
observation = Observation(
    images=images,
    image_masks=image_masks,
    state=state,
    tokenized_prompt=None,
    tokenized_prompt_mask=None,
)

# 运行模型推理
with torch.no_grad():
    model.eval()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 将观察对象移到设备上
    observation_device = Observation(
        images={k: v.to(device) for k, v in observation.images.items()},
        image_masks={k: v.to(device) for k, v in observation.image_masks.items()},
        state=observation.state.to(device),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
    )
    
    try:
        # 调用模型的推理方法
        actions = model.sample_actions(device, observation_device)
        
        print("Actions shape:", actions.shape)
        print("Actions:", actions)
        
    except Exception as e:
        print(f"Inference failed: {e}")
        print("Error details:", str(e))

print("Inference completed!")
