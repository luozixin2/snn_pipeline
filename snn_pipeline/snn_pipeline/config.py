# snn_pipeline/config.py
import os
import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class Config:
    # 神经元参数
    time_steps: int = 10
    tau_mem: float = 10.0
    tau_syn: float = 5.0
    threshold: float = 0.5
    reset_mode: str = "zero"  # "subtract" or "zero"
    
    # 流水线参数
    pipeline_width: int = 4
    parallel_blocks: int = 2
    
    # 内存优化
    bit_pack: bool = True
    bit_pack_dim: int = 32  # 打包位宽，通常为32位或64位
    use_memory_pool: bool = True
    max_memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 1e-3
    surrogate_gradient: str = "atan"  # "fast_sigmoid", "sigmoid", "triangle"
    surrogate_slope: float = 25.0
    
    # CUDA相关
    cuda_threads_per_block: int = 256
    max_cuda_streams: int = 4
    
    # 额外参数存储
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        config = cls()
        prefix = "SNN_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                if hasattr(config, config_key):
                    attr_type = type(getattr(config, config_key))
                    try:
                        setattr(config, config_key, attr_type(value))
                    except ValueError:
                        pass  # 忽略类型转换错误
        return config
    
    def update(self, **kwargs) -> "Config":
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value
        return self