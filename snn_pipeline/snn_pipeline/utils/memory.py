import torch
import torch.fx
from typing import Tuple, Dict, Any

def pack_spikes(spikes: torch.Tensor, pack_dim: int = 32) -> torch.Tensor:
    """
    将脉冲信号压缩为位压缩格式。

    在FX符号追踪（symbolic_trace）时，spikes是Proxy对象，不能进行值判断。
    因此，在FX模式下，跳过值检查，只进行形状变换和打包。

    Args:
        spikes: 脉冲张量，形状为[batch, ...]，值为0或1。
        pack_dim: 打包维度，默认为32。

    Returns:
        压缩后的整数张量。
    """

    is_proxy = isinstance(spikes, torch.fx.Proxy)

    B = spikes.shape[0]
    # flatten 除 batch 外所有 dims
    flat = spikes.reshape(B, -1)
    L = flat.shape[1]

    if not is_proxy:
        # 真实 Tensor 时补齐到 pack_dim 的整数倍
        rem = int(L % pack_dim)
        if rem != 0:
            pad = pack_dim - rem
            pad_tensor = torch.zeros(B, pad, dtype=spikes.dtype, device=spikes.device)
            flat = torch.cat([flat, pad_tensor], dim=1)
            L = flat.shape[1]

    # 重新 view 成 [B, L//pack_dim, pack_dim]；-1 让框架自动推 dim0
    reshaped = flat.reshape(B, -1, pack_dim)

    # 转 int32
    reshaped_int = reshaped.to(torch.int32)

    # 生成权重向量 [pack_dim]
    # 这里 pack_dim 是 Python int，arange 返回真值张量，不涉及 Proxy
    weights = torch.arange(pack_dim, dtype=torch.int32)    # 在 CPU 上
    weights = weights.to(spikes.device)                    # 再搬到目标设备
    # 广播相乘后 sum
    # reshaped_int: [B, M, pack_dim], weights: [pack_dim]
    packed = (reshaped_int * weights).sum(dim=-1)

    return packed

def unpack_spikes(packed_spikes: torch.Tensor, original_shape: Tuple[int, ...], 
                  pack_dim: int = 32) -> torch.Tensor:
    """
    将位压缩格式的信号解包为原始脉冲
    Args:
        packed_spikes: 压缩后的整数张量，形状为[batch, num_ints]
        original_shape: 原始形状，形状为(batch, ...)
        pack_dim: 打包维度，与pack_spikes保持一致
    Returns:
        解压后的脉冲张量
    """
    B, M = packed_spikes.shape
    # 解包到 [B, M, pack_dim]
    unpacked = torch.zeros(B, M, pack_dim,
                           dtype=torch.float32,
                           device=packed_spikes.device)
    for i in range(pack_dim):
        mask = 1 << i
        unpacked[:, :, i] = ((packed_spikes & mask) >> i).to(torch.float32)

    # 展平并截断
    flat_size = 1
    for d in original_shape[1:]:
        flat_size *= d
    flat = unpacked.reshape(B, -1)[:, :flat_size]

    return flat.reshape(original_shape)


class MemoryPool:
    """内存池，用于重用张量以减少内存碎片"""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024):
        self.max_size = max_size  # 池最大容量（字节）
        self.current_size = 0     # 当前使用容量
        self.pools = {}           # 按形状和数据类型分组的张量池
        
    def get(self, shape: Tuple[int, ...], dtype: torch.dtype, 
            device: torch.device) -> torch.Tensor:
        """
        从池中获取张量
        Args:
            shape: 张量形状
            dtype: 数据类型
            device: 设备
        Returns:
            张量
        """
        # 计算张量大小
        key = (shape, dtype, device)
        
        # 检查池中是否有可用张量
        if key in self.pools and self.pools[key]:
            # 从池中弹出一个张量
            tensor = self.pools[key].pop()
            # 如果池为空，删除该键
            if not self.pools[key]:
                del self.pools[key]
            return tensor.zero_()
        
        # 如果没有可用张量，创建新的
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def put(self, tensor: torch.Tensor) -> None:
        """
        将张量放回池中
        Args:
            tensor: 要放回的张量
        """
        # 计算张量大小
        tensor_size = tensor.numel() * tensor.element_size()
        key = (tensor.shape, tensor.dtype, tensor.device)
        
        # 检查是否超过最大容量
        if self.current_size + tensor_size > self.max_size:
            # 如果超过容量，不存储
            return
        
        # 将张量存入池
        if key not in self.pools:
            self.pools[key] = []
        self.pools[key].append(tensor.detach())
        self.current_size += tensor_size
    
    def clear(self) -> None:
        """清空内存池"""
        self.pools.clear()
        self.current_size = 0

# 全局内存池
_global_memory_pool = MemoryPool()

def get_tensor(shape: Tuple[int, ...], dtype: torch.dtype, 
               device: torch.device) -> torch.Tensor:
    """从全局内存池获取张量"""
    return _global_memory_pool.get(shape, dtype, device)

def recycle_tensor(tensor: torch.Tensor) -> None:
    """将张量回收到全局内存池"""
    _global_memory_pool.put(tensor)

def clear_memory_pool() -> None:
    """清空全局内存池"""
    _global_memory_pool.clear()