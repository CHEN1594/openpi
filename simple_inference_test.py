#!/usr/bin/env python3
"""
简单的推理测试脚本，测试我们训练的π0.5模型
"""
import os
import sys
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append('/data/home/zhangjing2/th/openpi/src')

from openpi.training import config as _config
from openpi.policies import policy_config

def test_model_inference():
    """测试模型推理"""
    print("🚀 开始测试π0.5模型推理...")
    
    # 加载配置
    config = _config.get_config("pi05_libero")
    checkpoint_dir = "/data/home/zhangjing2/th/openpi/checkpoints/pi05_libero/pytorch_libero_4gpu/24000"
    
    print(f"📂 检查点目录: {checkpoint_dir}")
    print(f"✅ 检查点存在: {os.path.exists(checkpoint_dir)}")
    
    try:
        # 创建策略
        print("🔄 正在加载模型...")
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        print("✅ 模型加载成功!")
        
        # 创建测试数据
        print("🔄 创建测试数据...")
        
        # 创建虚拟图像数据 (224x224 RGB)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_wrist_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 创建虚拟状态数据 (8维)
        dummy_state = np.random.randn(8).astype(np.float32)
        
        # 创建测试输入
        test_input = {
            "observation/image": dummy_image,
            "observation/wrist_image": dummy_wrist_image,
            "observation/state": dummy_state,
            "prompt": "pick up the red block"
        }
        
        print("🔄 运行推理...")
        result = policy.infer(test_input)
        
        print("✅ 推理成功!")
        print(f"📊 输出动作形状: {result['actions'].shape}")
        print(f"📊 动作范围: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
        print(f"📊 动作均值: {result['actions'].mean():.3f}")
        print(f"📊 动作标准差: {result['actions'].std():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_inference()
    if success:
        print("\n🎉 模型推理测试成功!")
    else:
        print("\n💥 模型推理测试失败!")
