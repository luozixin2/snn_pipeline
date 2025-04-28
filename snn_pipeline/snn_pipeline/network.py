# snn_pipeline/network.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union, Callable, Type
from .layers import SpikingLayer, InputEncoder, OutputDecoder
from .config import Config

class SpikingModule(nn.Module):
    """脉冲神经网络模块基类，类似于torch.nn.Module"""
    
    def __init__(self, time_steps: int = 10, config: Optional[Config] = None):
        super().__init__()
        self.config = config or Config()
        self.time_steps = time_steps
        self.layers = nn.ModuleList()
        self.input_encoder = None
        self.output_decoder = None
        
    def add_layer(self, layer: nn.Module) -> 'SpikingModule':
        """添加层"""
        self.layers.append(layer)
        return self
    
    def add_encoder(self, encoder: InputEncoder) -> 'SpikingModule':
        """添加输入编码器"""
        self.input_encoder = encoder
        return self
    
    def add_decoder(self, decoder: OutputDecoder) -> 'SpikingModule':
        """添加输出解码器"""
        self.output_decoder = decoder
        return self
    
    def forward(self, x: torch.Tensor, time_steps: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量
            time_steps: 时间步数，如果为None则使用默认值
        Returns:
            输出张量
        """
        t_steps = time_steps or self.time_steps
        
        # 输入编码
        if self.input_encoder is not None:
            spike_inputs = self.input_encoder(x)
        else:
            # 默认直接复制
            spike_inputs = [x for _ in range(t_steps)]
        
        # 重置所有层的状态
        self.reset_states()
        
        # 按时间步进行前向传播
        spike_outputs = []
        for t in range(t_steps):
            x_t = spike_inputs[t]
            for layer in self.layers:
                x_t = layer(x_t)
            spike_outputs.append(x_t)
        
        # 输出解码
        if self.output_decoder is not None:
            return self.output_decoder(spike_outputs)
        else:
            # 默认返回平均发放率
            return torch.stack(spike_outputs).mean(dim=0)
    
    def reset_states(self) -> None:
        """重置所有层的状态"""
        for layer in self.layers:
            if isinstance(layer, SpikingLayer):
                layer.reset_state()
    
    def detach_states(self) -> None:
        """分离所有层的状态梯度"""
        for layer in self.layers:
            if isinstance(layer, SpikingLayer):
                layer.detach_state()


class Sequential(SpikingModule):
    """脉冲序列模型，类似于torch.nn.Sequential"""
    
    def __init__(self, *layers, time_steps: int = 10, config: Optional[Config] = None):
        super().__init__(time_steps, config)
        for layer in layers:
            self.add_layer(layer)


class SpikingResidualBlock(SpikingModule):
    """脉冲残差块"""
    
    def __init__(self, main_path: nn.ModuleList, shortcut_path: Optional[nn.ModuleList] = None,
                 time_steps: int = 10, config: Optional[Config] = None):
        super().__init__(time_steps, config)
        self.main_path = main_path
        self.shortcut_path = shortcut_path or nn.Identity()
        
    def forward(self, x: torch.Tensor, time_steps: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量
            time_steps: 时间步数，如果为None则使用默认值
        Returns:
            输出张量
        """
        t_steps = time_steps or self.time_steps
        
        # 输入编码
        if self.input_encoder is not None:
            spike_inputs = self.input_encoder(x)
        else:
            # 默认直接复制
            spike_inputs = [x for _ in range(t_steps)]
        
        # 重置所有层的状态
        self.reset_states()
        
        # 按时间步进行前向传播
        spike_outputs = []
        for t in range(t_steps):
            x_t = spike_inputs[t]
            
            # 主路径
            main_out = x_t
            for layer in self.main_path:
                main_out = layer(main_out)
            
            # 捷径路径
            shortcut_out = self.shortcut_path(x_t)
            
            # 残差连接
            x_t = main_out + shortcut_out
            spike_outputs.append(x_t)
        
        # 输出解码
        if self.output_decoder is not None:
            return self.output_decoder(spike_outputs)
        else:
            # 默认返回平均发放率
            return torch.stack(spike_outputs).mean(dim=0)


class ModelFactory:
    """模型工厂，用于创建预定义的脉冲神经网络模型"""
    
    @staticmethod
    def create_snn_mlp(input_size: int, hidden_sizes: List[int], output_size: int, 
                        config: Optional[Config] = None) -> SpikingModule:
        """
        创建多层感知机SNN
        Args:
            input_size: 输入大小
            hidden_sizes: 隐藏层大小列表
            output_size: 输出大小
            config: 配置对象
        Returns:
            SpikingModule实例
        """
        from .layers import SpikingLinear
        
        config = config or Config()
        model = Sequential(time_steps=config.time_steps, config=config)
        
        # 添加输入编码器
        model.add_encoder(InputEncoder((input_size,), "rate", config.time_steps))
        
        # 添加隐藏层
        last_size = input_size
        for hidden_size in hidden_sizes:
            model.add_layer(SpikingLinear(last_size, hidden_size, True, "LIF", 
                                          tau_mem=config.tau_mem,
                                          threshold=config.threshold,
                                          reset_mode=config.reset_mode,
                                          surrogate_gradient=config.surrogate_gradient,
                                          surrogate_slope=config.surrogate_slope))
            last_size = hidden_size
        
        # 添加输出层
        model.add_layer(SpikingLinear(last_size, output_size, True, "LIF",
                                      tau_mem=config.tau_mem,
                                      threshold=config.threshold,
                                      reset_mode=config.reset_mode,
                                      surrogate_gradient=config.surrogate_gradient,
                                      surrogate_slope=config.surrogate_slope))
        
        # 添加输出解码器
        model.add_decoder(OutputDecoder("rate"))
        
        return model
    
    @staticmethod
    def create_snn_cnn(input_channels: int, input_size: Tuple[int, int], 
                       conv_configs: List[Dict[str, Any]], fc_sizes: List[int], 
                       output_size: int, config: Optional[Config] = None) -> SpikingModule:
        """
        创建卷积SNN
        Args:
            input_channels: 输入通道数
            input_size: 输入大小 (H, W)
            conv_configs: 卷积层配置列表，每个元素是包含out_channels、kernel_size等的字典
            fc_sizes: 全连接层大小列表
            output_size: 输出大小
            config: 配置对象
        Returns:
            SpikingModule实例
        """
        from .layers import FlattenLayer, SpikingConv2d, SpikingMaxPool2d, SpikingLinear
        
        config = config or Config()
        model = Sequential(time_steps=config.time_steps, config=config)
        
        # 添加输入编码器
        model.add_encoder(InputEncoder((input_channels, *input_size), "rate", config.time_steps))
        
        # 计算卷积后的特征大小
        curr_channels = input_channels
        curr_size = list(input_size)
        
        # 添加卷积层
        for i, conv_config in enumerate(conv_configs):
            out_channels = conv_config.get("out_channels", 32)
            kernel_size = conv_config.get("kernel_size", 3)
            stride = conv_config.get("stride", 1)
            padding = conv_config.get("padding", 0)
            pool_size = conv_config.get("pool_size", 2)
            
            # 添加卷积层
            model.add_layer(SpikingConv2d(
                curr_channels, out_channels, kernel_size, stride, padding,
                neuron_model="LIF",
                tau_mem=config.tau_mem,
                threshold=config.threshold,
                reset_mode=config.reset_mode,
                surrogate_gradient=config.surrogate_gradient,
                surrogate_slope=config.surrogate_slope
            ))
            
            # 更新特征大小
            curr_channels = out_channels
            curr_size[0] = (curr_size[0] + 2 * padding - kernel_size) // stride + 1
            curr_size[1] = (curr_size[1] + 2 * padding - kernel_size) // stride + 1
            
            # 添加池化层
            if pool_size > 1:
                model.add_layer(SpikingMaxPool2d(
                    pool_size, pool_size, 0,
                    neuron_model="LIF",
                    tau_mem=config.tau_mem,
                    threshold=config.threshold,
                    reset_mode=config.reset_mode,
                    surrogate_gradient=config.surrogate_gradient,
                    surrogate_slope=config.surrogate_slope
                ))
                
                # 更新特征大小
                curr_size[0] = curr_size[0] // pool_size
                curr_size[1] = curr_size[1] // pool_size
        
        # 特征平铺
        flatten_size = curr_channels * curr_size[0] * curr_size[1]
        
        # 添加全连接层
        last_size = flatten_size
        model.add_layer(FlattenLayer)
        for fc_size in fc_sizes:
            model.add_layer(SpikingLinear(
                last_size, fc_size, True,
                neuron_model="LIF",
                tau_mem=config.tau_mem,
                threshold=config.threshold,
                reset_mode=config.reset_mode,
                surrogate_gradient=config.surrogate_gradient,
                surrogate_slope=config.surrogate_slope
            ))
            last_size = fc_size
        
        # 添加输出层
        model.add_layer(SpikingLinear(
            last_size, output_size, True,
            neuron_model="LIF",
            tau_mem=config.tau_mem,
            threshold=config.threshold,
            reset_mode=config.reset_mode,
            surrogate_gradient=config.surrogate_gradient,
            surrogate_slope=config.surrogate_slope
        ))
        
        # 添加输出解码器
        model.add_decoder(OutputDecoder("rate"))
        
        return model