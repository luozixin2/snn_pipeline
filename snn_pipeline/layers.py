# snn_pipeline/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Union, Type
from .neurons import BaseNeuron, LIFNeuron, IzhikevichNeuron
from .config import Config

class SpikingLayer(nn.Module):
    """脉冲层基类"""
    
    def __init__(self, neuron_model: Union[str, Type[BaseNeuron]] = "LIF", **neuron_params):
        super().__init__()
        # >>> 1. 从 Config 拉一份默认值 <<<
        default = {
            'tau_mem':    Config.tau_mem,
            'tau_syn':    Config.tau_syn,
            'threshold':  Config.threshold,
            'reset_mode': Config.reset_mode,
            'bit_pack': Config.bit_pack,
            'surrogate_gradient': Config.surrogate_gradient,
            'surrogate_slope': Config.surrogate_slope,
        }
        # >>> 2. 用外面传入的参数覆盖默认值 <<<
        merged = {**default, **neuron_params}
        # 神经元类型
        if isinstance(neuron_model, str):
            if neuron_model == "LIF":
                self.neuron = LIFNeuron(**merged)
            elif neuron_model == "Izhikevich":
                self.neuron = IzhikevichNeuron(**merged)
            else:
                raise ValueError(f"Unknown neuron model: {neuron_model}")
        else:
            self.neuron = neuron_model(**merged)
        
        self.requires_init_state = True
        self.state = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回脉冲输出"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def reset_state(self) -> None:
        """重置神经元状态"""
        if self.state is not None:
            self.state = self.neuron.reset(self.state)
        self.requires_init_state = True
    
    def detach_state(self) -> None:
        """分离状态梯度"""
        if self.state is not None:
            for key in self.state:
                if isinstance(self.state[key], torch.Tensor):
                    self.state[key] = self.state[key].detach()


class SpikingConv2d(SpikingLayer):
    """脉冲卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True,
                 neuron_model: Union[str, Type[BaseNeuron]] = "LIF", **neuron_params):
        super().__init__(neuron_model, **neuron_params)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # 输出形状会在前向传播中计算
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回脉冲输出"""
        x = self.conv(x)
        if self.requires_init_state or self.state is None:
            self.state = self.neuron.init_state(x)
            self.requires_init_state = False
            
        spike, self.state = self.neuron.forward(x, self.state)
        return spike


class SpikingAvgPool2d(SpikingLayer):
    """脉冲平均池化层"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 neuron_model: Union[str, Type[BaseNeuron]] = "LIF", **neuron_params):
        super().__init__(neuron_model, **neuron_params)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)
        # 输出形状会在前向传播中计算
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回脉冲输出"""
        x = self.pool(x)
        
        if self.requires_init_state or self.state is None:
            output_shape = (x.shape[1], x.shape[2], x.shape[3])  # (C, H, W)
            self.state = self.neuron.init_state(x.shape[0], output_shape)
            self.requires_init_state = False
            
        spike, self.state = self.neuron.forward(x, self.state)
        return spike


class SpikingMaxPool2d(SpikingLayer):
    """脉冲最大池化层"""
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 neuron_model: Union[str, Type[BaseNeuron]] = "LIF", **neuron_params):
        super().__init__(neuron_model, **neuron_params)
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        # 输出形状会在前向传播中计算
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回脉冲输出"""
        x = self.pool(x)
        
        if self.requires_init_state or self.state is None:
            self.state = self.neuron.init_state(x)
            self.requires_init_state = False
            
        spike, self.state = self.neuron.forward(x, self.state)
        return spike


class SpikingBatchNorm2d(SpikingLayer):
    """脉冲批归一化层"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 neuron_model: Union[str, Type[BaseNeuron]] = "LIF", **neuron_params):
        super().__init__(neuron_model, **neuron_params)
        self.bn = nn.BatchNorm2d(num_features, eps, momentum)
        # 输出形状会在前向传播中计算
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回脉冲输出"""
        x = self.bn(x)
        
        if self.requires_init_state or self.state is None:
            self.state = self.neuron.init_state(x)
            self.requires_init_state = False
            
        spike, self.state = self.neuron.forward(x, self.state)
        return spike


class InputEncoder(nn.Module):
    """输入编码器，将常规输入转换为脉冲序列"""
    
    def __init__(self, input_shape: Tuple[int, ...], encoding: str = "rate", time_steps: int = 10):
        super().__init__()
        self.input_shape = input_shape
        self.encoding = encoding
        self.time_steps = time_steps
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """将输入转换为T个时间步长的脉冲序列"""
        batch_size = x.shape[0]
        
        if self.encoding == "rate":
            # 转换为发放率编码
            x_repeated = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
            # 改用 rand_like
            random = torch.rand_like(x_repeated)
            spikes = (random < x_repeated).float()
            return torch.stack([spikes[t] for t in range(self.time_steps)],dim=0)
            
        elif self.encoding == "temporal":
            # 时间编码 - 依据值大小决定发放时间
            # 值越大，越早发放脉冲
            spikes = []
            for t in range(self.time_steps):
                # 归一化阈值，随时间线性增加
                threshold = (t + 1) / self.time_steps
                spike_t = (x >= threshold).float()
                # 确保每个位置只发放一次
                if t > 0:
                    # 减去之前所有时间步已发放的
                    for prev_t in range(t):
                        spike_t = spike_t * (1 - spikes[prev_t])
                spikes.append(spike_t)
            return torch.stack(spikes,dim=0)
            
        elif self.encoding == "direct":
            # 直接复制 - 用于已编码的输入
            return torch.stack([x for _ in range(self.time_steps)],dim=0)
            
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")


class OutputDecoder(nn.Module):
    """输出解码器，将脉冲序列转换为常规输出"""
    
    def __init__(self, decoding: str = "rate"):
        super().__init__()
        self.decoding = decoding
        
    def forward(self, spike_seq: List[torch.Tensor]) -> torch.Tensor:
        """将脉冲序列转换为单个输出"""
        if self.decoding == "rate":
            # 计算平均发放率
            return torch.stack(spike_seq,dim=0).mean(dim=0)
            
        elif self.decoding == "latency":
            # 第一个脉冲的时间
            time_steps = len(spike_seq)
            batch_size = spike_seq[0].shape[0]
            output_shape = spike_seq[0].shape[1:]
            
            # 初始化最大时间步
            first_spike = torch.ones((batch_size, *output_shape), device=spike_seq[0].device) * time_steps
            
            # 找到第一个脉冲的时间
            for t, spike in enumerate(spike_seq):
                # 更新尚未发放脉冲的位置
                mask = (first_spike >= time_steps).float()
                first_spike = first_spike * (1 - spike * mask) + spike * mask * t
                
            # 归一化时间
            return 1.0 - (first_spike / time_steps)
            
        elif self.decoding == "max":
            # 取最大值
            return torch.stack(spike_seq).max(dim=0)[0]
            
        elif self.decoding == "sum":
            # 累积所有脉冲
            return sum(spike_seq)
            
        else:
            raise ValueError(f"Unknown decoding: {self.decoding}")
        

class FlattenLayer(nn.Module):
    """把 [batch, *] 展平为 [batch, -1]"""
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(self.start_dim, self.end_dim)

class ReshapeLayer(nn.Module):
    """
    把输入 reshape 到指定形状。
    target_shape 可以省略 batch 维度，如 (C*H*W,)
    """
    def __init__(self, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保留 batch 维度不动
        batch = x.shape[0]
        return x.view((batch, *self.target_shape))

class PermuteLayer(nn.Module):
    """
    对输入做维度重排。
    dims: 一个整数元组，代表 permute 的维度顺序（包括 batch 维）。
    例如想把 [B, C, H, W] → [B, H, W, C]，请传 (0,2,3,1)
    """
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)