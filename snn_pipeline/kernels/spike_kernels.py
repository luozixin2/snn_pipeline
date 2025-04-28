# snn_pipeline/kernels/spike_kernels.py
import torch
import torch.nn.functional as F
from typing import Tuple, Any, Optional, Union
from torch.utils.cpp_extension import load

# 这里应当加载自定义CUDA扩展
# 在实际实现中，你需要编译spike_kernels.cu文件
# 为简化示例，这里仅使用PyTorch函数模拟

"""
尝试加载CUDA扩展
如果编译好了就使用，否则使用Python实现
"""
try:
    # 加载已编译的CUDA扩展
    spike_cuda = load(
        name="spike_cuda",
        sources=["snn_pipeline/kernels/spike_kernels.cu"],
        verbose=True
    )
    _CUDA_EXTENSION_LOADED = True
except:
    _CUDA_EXTENSION_LOADED = False
    print("CUDA extension not loaded, using PyTorch implementation instead")


class SpikeFunctionCPU(torch.autograd.Function):
    """CPU实现的脉冲函数"""
    
    @staticmethod
    def forward(ctx, membrane_potential, threshold, reset_mode="subtract"):
        """
        前向传播
        Args:
            membrane_potential: 膜电位
            threshold: 阈值
            reset_mode: 重置模式，"subtract"或"zero"
        Returns:
            脉冲
        """
        spike = (membrane_potential > threshold).float()
        
        if reset_mode == "subtract":
            reset_potential = membrane_potential - spike * threshold
        elif reset_mode == "zero":
            reset_potential = membrane_potential * (1 - spike)
        else:
            raise ValueError(f"Unknown reset mode: {reset_mode}")
        
        # 保存上下文
        ctx.save_for_backward(membrane_potential, threshold)
        ctx.reset_mode = reset_mode
        
        return spike, reset_potential
    
    @staticmethod
    def backward(ctx, grad_spike, grad_reset_potential):
        """
        反向传播
        Args:
            grad_spike: 脉冲梯度
            grad_reset_potential: 重置电位梯度
        Returns:
            膜电位梯度
        """
        membrane_potential, threshold = ctx.saved_tensors
        reset_mode = ctx.reset_mode
        
        # 使用代理梯度
        grad_surrogate = 0.3 * torch.exp(-0.05 * abs(membrane_potential - threshold)) / \
                        (torch.exp(-0.05 * abs(membrane_potential - threshold)) + 1) ** 2
        
        # 计算膜电位梯度
        if reset_mode == "subtract":
            grad_membrane = grad_reset_potential + grad_spike * grad_surrogate
        elif reset_mode == "zero":
            grad_membrane = grad_reset_potential * (1 - (membrane_potential > threshold).float()) + \
                          grad_spike * grad_surrogate
        
        return grad_membrane, None, None


class SpikeFunctionCUDA(torch.autograd.Function):
    """CUDA实现的脉冲函数"""
    
    @staticmethod
    def forward(ctx, membrane_potential, threshold, reset_mode="subtract"):
        """
        前向传播
        Args:
            membrane_potential: 膜电位
            threshold: 阈值
            reset_mode: 重置模式，"subtract"或"zero"
        Returns:
            脉冲
        """
        if _CUDA_EXTENSION_LOADED:
            spike, reset_potential = spike_cuda.forward(
                membrane_potential, threshold, reset_mode
            )
        else:
            # 回退到CPU实现
            spike, reset_potential = SpikeFunctionCPU.apply(
                membrane_potential, threshold, reset_mode
            )
        
        # 保存上下文
        ctx.save_for_backward(membrane_potential, threshold)
        ctx.reset_mode = reset_mode
        
        return spike, reset_potential
    
    @staticmethod
    def backward(ctx, grad_spike, grad_reset_potential):
        """
        反向传播
        Args:
            grad_spike: 脉冲梯度
            grad_reset_potential: 重置电位梯度
        Returns:
            膜电位梯度
        """
        membrane_potential, threshold = ctx.saved_tensors
        reset_mode = ctx.reset_mode
        
        if _CUDA_EXTENSION_LOADED:
            grad_membrane = spike_cuda.backward(
                grad_spike, grad_reset_potential, 
                membrane_potential, threshold, reset_mode
            )
        else:
            # 回退到CPU实现
            grad_surrogate = 0.3 * torch.exp(-0.05 * abs(membrane_potential - threshold)) / \
                            (torch.exp(-0.05 * abs(membrane_potential - threshold)) + 1) ** 2
            
            # 计算膜电位梯度
            if reset_mode == "subtract":
                grad_membrane = grad_reset_potential + grad_spike * grad_surrogate
            elif reset_mode == "zero":
                grad_membrane = grad_reset_potential * (1 - (membrane_potential > threshold).float()) + \
                              grad_spike * grad_surrogate
        
        return grad_membrane, None, None


# 根据是否可用CUDA选择实现
SpikeFunction = SpikeFunctionCUDA if torch.cuda.is_available() else SpikeFunctionCPU

def spike_kernel_forward(membrane_potential: torch.Tensor, threshold: float, 
                        reset_mode: str = "subtract") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    脉冲前向传播核心函数
    Args:
        membrane_potential: 膜电位
        threshold: 阈值
        reset_mode: 重置模式，"subtract"或"zero"
    Returns:
        脉冲和重置后的膜电位
    """
    return SpikeFunction.apply(membrane_potential, threshold, reset_mode)

def spike_kernel_backward(grad_spike: torch.Tensor, grad_reset_potential: torch.Tensor,
                         membrane_potential: torch.Tensor, threshold: float, 
                         reset_mode: str = "subtract") -> torch.Tensor:
    """
    脉冲反向传播核心函数（手动调用）
    Args:
        grad_spike: 脉冲梯度
        grad_reset_potential: 重置电位梯度
        membrane_potential: 膜电位
        threshold: 阈值
        reset_mode: 重置模式，"subtract"或"zero"
    Returns:
        膜电位梯度
    """
    # 使用代理梯度
    grad_surrogate = 0.3 * torch.exp(-0.05 * abs(membrane_potential - threshold)) / \
                    (torch.exp(-0.05 * abs(membrane_potential - threshold)) + 1) ** 2
    
    # 计算膜电位梯度
    if reset_mode == "subtract":
        grad_membrane = grad_reset_potential + grad_spike * grad_surrogate
    elif reset_mode == "zero":
        grad_membrane = grad_reset_potential * (1 - (membrane_potential > threshold).float()) + \
                      grad_spike * grad_surrogate
    
    return grad_membrane