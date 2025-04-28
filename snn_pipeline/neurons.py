# snn_pipeline/neurons.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, Callable, Union
from .utils.memory import pack_spikes, unpack_spikes
from .kernels.spike_kernels import spike_kernel_forward, spike_kernel_backward


class SurrogateGradient:
    """代理梯度函数集合"""

    @staticmethod
    def fast_sigmoid(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
        z = slope * x
        z = torch.clamp(z, min=-10.0, max=10.0)
        return 1.0 / (1.0 + torch.exp(-z))

    @staticmethod
    def sigmoid(x: torch.Tensor, slope: float = 1.0) -> torch.Tensor:
        z = slope * x
        z = torch.clamp(z, min=-10.0, max=10.0)
        return torch.sigmoid(z)

    @staticmethod
    def triangle(x: torch.Tensor, slope: float = 1.0) -> torch.Tensor:
        return torch.clamp(slope * (1.0 - torch.abs(x)), min=0.0)

    @staticmethod
    def arctan(x: torch.Tensor, slope: float = 1.0) -> torch.Tensor:
        """
        ATan surrogate：没有 exp，数值更稳定。
        grad = d/dx arctan(slope * x) = slope / (1 + (slope*x)^2)
        我们直接返回它做为 surrogate gradient。
        """
        z = slope * x
        return slope / (1.0 + z * z)

    @classmethod
    def get(cls, name: str) -> Callable:
        """获取代理梯度函数"""
        if name == "fast_sigmoid":
            return cls.fast_sigmoid
        elif name == "sigmoid":
            return cls.sigmoid
        elif name == "triangle":
            return cls.triangle
        elif name in ("atan", "arctan"):
            return cls.arctan
        else:
            raise ValueError(f"Unknown surrogate gradient: {name!r}")


class BaseNeuron(nn.Module):
    """神经元基类，定义通用接口"""

    def __init__(self,
                 threshold: float = 1.0,
                 reset_mode: str = "subtract",
                 surrogate_gradient: str = "fast_sigmoid",
                 surrogate_slope: float = 25.0,
                 bit_pack: bool = False,
                 bit_pack_dim: int = 32):
        super().__init__()
        self.threshold = threshold
        self.reset_mode = reset_mode
        self.surrogate_fn = SurrogateGradient.get(surrogate_gradient)
        self.surrogate_slope = surrogate_slope
        self.bit_pack = bit_pack
        self.bit_pack_dim = bit_pack_dim

    def forward(self, x: torch.Tensor,
                state: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播，返回脉冲输出和更新后的状态"""
        raise NotImplementedError("Subclasses must implement forward method")

    def init_state(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """初始化神经元状态"""
        raise NotImplementedError("Subclasses must implement init_state method")

    def reset(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """重置神经元状态"""
        raise NotImplementedError("Subclasses must implement reset method")


class LIFNeuron(BaseNeuron):
    """Leaky Integrate-and-Fire 神经元"""

    def __init__(self,
                 tau_mem: float = 10.0,
                 tau_syn: Optional[float] = None,
                 threshold: float = 1.0,
                 reset_mode: str = "subtract",
                 surrogate_gradient: str = "fast_sigmoid",
                 surrogate_slope: float = 25.0,
                 bit_pack: bool = False,
                 bit_pack_dim: int = 32):
        super().__init__(threshold, reset_mode,
                         surrogate_gradient, surrogate_slope,
                         bit_pack, bit_pack_dim)
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn if tau_syn is not None else tau_mem
        self.decay_mem = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.decay_syn = torch.exp(torch.tensor(-1.0 / tau_syn))

    def forward(self,
                x: torch.Tensor,
                state: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state is None:
            # 传入的 init_state 只接受一个 Tensor
            state = self.init_state(x)

        # 获取并 detach 上一时刻的状态，切断前一次计算图
        mem = state["membrane"].detach()
        syn = state["synaptic"].detach()

        # 更新突触电流
        syn = self.decay_syn * syn + x

        # 更新膜电位
        mem = self.decay_mem * mem + syn

        # 生成脉冲（硬阈值）
        spike = (mem > self.threshold).float()

        # 重置膜电位
        if self.reset_mode == "subtract":
            mem = mem - spike * self.threshold
        elif self.reset_mode == "zero":
            mem = mem * (1 - spike)

        # surrogate 前向后向分离
        spike_grad = self.surrogate_fn(mem - self.threshold, self.surrogate_slope)
        spike = spike.detach() + (spike_grad - spike_grad.detach())

        # 可选：位打包
        if self.bit_pack:
            spike_packed = pack_spikes(spike, self.bit_pack_dim)
            state.update({
                "membrane": mem,
                "synaptic": syn,
                "spike_packed": spike_packed
            })
            return spike, state

        # 更新状态
        state.update({
            "membrane": mem,
            "synaptic": syn
        })
        return spike, state

    def init_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """初始化神经元状态"""
        z = torch.zeros_like(x)
        return {"membrane": z, "synaptic": z.clone()}

    def reset(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """重置神经元状态"""
        state["membrane"].zero_()
        state["synaptic"].zero_()
        return state


class IzhikevichNeuron(BaseNeuron):
    """Izhikevich 神经元模型"""

    def __init__(self,
                 a: float = 0.02,
                 b: float = 0.2,
                 c: float = -65.0,
                 d: float = 8.0,
                 threshold: float = 30.0,
                 reset_mode: str = "custom",
                 surrogate_gradient: str = "fast_sigmoid",
                 surrogate_slope: float = 25.0,
                 bit_pack: bool = False,
                 bit_pack_dim: int = 32):
        super().__init__(threshold, reset_mode,
                         surrogate_gradient, surrogate_slope,
                         bit_pack, bit_pack_dim)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self,
                x: torch.Tensor,
                state: Optional[Dict[str, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if state is None:
            state = self.init_state(x)

        # 获取并 detach 上一时刻的状态
        v = state["voltage"].detach()
        u = state["recovery"].detach()

        # 更新膜电位和恢复变量（两步更新提高精度）
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + x)
        v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + x)
        u = u + self.a * (self.b * v - u)

        # 生成脉冲
        spike = (v >= self.threshold).float()

        # 重置
        v = v * (1 - spike) + spike * self.c
        u = u + spike * self.d

        # surrogate 前向后向分离
        spike_grad = self.surrogate_fn(v - self.threshold, self.surrogate_slope)
        spike = spike.detach() + (spike_grad - spike_grad.detach())

        # 位打包可选
        if self.bit_pack:
            spike_packed = pack_spikes(spike, self.bit_pack_dim)
            state.update({
                "voltage": v,
                "recovery": u,
                "spike_packed": spike_packed
            })
            return spike, state

        # 更新状态
        state.update({
            "voltage": v,
            "recovery": u
        })
        return spike, state

    def init_state(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """初始化神经元状态"""
        zeros = torch.zeros_like(x)
        volts = torch.full_like(x, fill_value=self.c)
        return {"voltage": volts, "recovery": zeros}

    def reset(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state["voltage"].fill_(self.c)
        state["recovery"].zero_()
        return state