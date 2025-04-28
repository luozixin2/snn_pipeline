# snn_pipeline/engine.py
import torch
import torch.cuda as cuda
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from .graph import ComputationGraph, GraphNode, GraphBlock
from .config import Config

class PipelineStage:
    """流水线阶段，包含一组可并行执行的操作"""
    
    def __init__(self, stage_id: int, nodes: List[GraphNode], stream: Optional[torch.cuda.Stream] = None):
        self.stage_id = stage_id
        self.nodes = nodes
        self.stream = stream or torch.cuda.Stream()
        self.inputs = {}  # 阶段输入缓存
        self.outputs = {}  # 阶段输出缓存
        
    def execute(self) -> None:
        """在指定 stream 上执行该阶段所有操作"""
        # 我们在每个 Stage 里，都维护一个 working buffer data，
        # 它最开始等同于 self.inputs，然后每节点算完就合并 outputs。
        data = dict(self.inputs)      # shallow copy
        # —— 新增：把所有输入 Tensor 告诉 Autograd，它们即将在 self.stream 上被用到
        for v in data.values():
            if isinstance(v, torch.Tensor):
                v.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            for node in self.nodes:
                try:
                    # 这里让 node 从 data 拿输入，结果写到 self.outputs
                    node.execute(data, self.outputs)
                    data.update(self.outputs)    # 把刚出炉的 outputs 也加回 data
                except KeyError as e:
                    print("=== PipelineStage.execute 崩溃 ===")
                    print(f" stage_id       = {self.stage_id}")
                    print(f" node.op_type   = {node.op_type}")
                    print(f" node.input_keys= {node.input_keys}")
                    print(f" 当前 data      = {list(data.keys())}")
                    print(f" 当前 outputs   = {list(self.outputs.keys())}")
                    raise
                # 把刚刚这个 node 的所有 outputs，合并回 data
                data.update(self.outputs)
        # 最后再把累积结果同步回 self.outputs（可选）
        # self.outputs.update(data)
        # PipelineStage.execute() 最后，或者 forward 后：

    
    def wait_for(self, other_stage: 'PipelineStage') -> None:
        """等待另一个阶段完成"""
        self.stream.wait_stream(other_stage.stream)


class PipelineEngine:
    """脉冲神经网络流水线执行引擎"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.graph = None
        self.stages = []
        self.is_built = False
        self.forward_stages = []
        self.backward_stages = []
        
    def build_pipeline(self, model: torch.nn.Module, sample_input: torch.Tensor) -> 'PipelineEngine':
        """
        构建流水线执行计划
        Args:
            model: 脉冲神经网络模型
            sample_input: 示例输入，用于推断形状
        Returns:
            self
        """
        # 创建计算图
        self.graph = ComputationGraph()
        # 构建前向计算图
        self.graph.build_forward_graph(model, sample_input, self.config.time_steps)
        
        # 将计算图分割为流水线阶段
        self._build_forward_stages()
        
        # 构建反向传播阶段（按原始顺序）
        # if self.config.pipeline_width > 1:  # 只有在实际使用流水线时才需要
        #     self.graph.build_backward_graph()
        #     self._build_backward_stages()
        
        self.is_built = True
        return self
    
    def _build_forward_stages(self) -> None:
        """构建前向传播流水线阶段"""
        # 根据配置决定每个阶段包含多少个计算块
        blocks_per_stage = max(1, len(self.graph.forward_blocks) // self.config.pipeline_width)
        
        # 创建CUDA流
        num_streams = min(self.config.max_cuda_streams, self.config.pipeline_width)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # 将计算块分配到阶段
        self.forward_stages = []
        for i in range(0, len(self.graph.forward_blocks), blocks_per_stage):
            stage_blocks = self.graph.forward_blocks[i:i+blocks_per_stage]
            stage_nodes = []
            for block in stage_blocks:
                stage_nodes.extend(block.nodes)
            
            stream_idx = len(self.forward_stages) % len(streams)
            stage = PipelineStage(len(self.forward_stages), stage_nodes, streams[stream_idx])
            self.forward_stages.append(stage)
    
    def _build_backward_stages(self) -> None:
        """构建反向传播阶段"""
        # 为简单起见，反向传播不使用流水线，按原始顺序执行
        # 每个反向节点一个阶段
        self.backward_stages = []
        backward_stream = torch.cuda.Stream()
        
        for i, node in enumerate(self.graph.backward_nodes):
            stage = PipelineStage(i, [node], backward_stream)
            self.backward_stages.append(stage)
    
    def forward(self, x: torch.Tensor, time_steps: Optional[int] = None) -> torch.Tensor:
        """
        流水线前向传播
        Args:
            x: 输入张量
            time_steps: 时间步数
        Returns:
            输出张量
        """
        if not self.is_built:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        t_steps = time_steps or self.config.time_steps

        # 一次性创建 streams, stages 已在 build_pipeline 里准备好
        default_stream = torch.cuda.current_stream()
        for t in range(t_steps):
            # 1) 为这个 time‐step 初始化一个全局 data 缓存
            data = {"input": x, "time_step": t}
            prev_stage = None

            # 2) 按 pipeline 顺序依次执行，每算完一个 stage 就合并它的 outputs
            for i, stage in enumerate(self.forward_stages):
                if prev_stage is not None:
                    stage.wait_for(prev_stage)
                stage.inputs = data            # 直接引用当前缓存
                stage.outputs = {}             # 清空上次残留

                stage.execute()                # 执行本 Stage 所有 node
                # —— 新增：给所有刚产生的 Tensor 标记到 default_stream
                for v in stage.outputs.values():
                    if isinstance(v, torch.Tensor):
                        v.record_stream(default_stream)

                # default_stream 等待这个 stage 的 stream 完成
                default_stream.wait_stream(stage.stream)

                data.update(stage.outputs)     # 立刻把新输出放回 data

                prev_stage = stage

            # 3) (可选) 如果你想在每个 time‐step 收集某些中间量，也可以在这里读 data

        # （可选）再次全局同步，保证所有 GPU op 都结束
        torch.cuda.synchronize()
        # 最后取 data 中的 key "output"（或你的模型定义里最后一个 op 写的 key）
        return data.get("output", None)
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        流水线反向传播
        Args:
            loss: 损失张量
        """
        if not self.is_built or not self.backward_stages:
            # 如果没有构建反向阶段，直接使用PyTorch自动微分
            loss.backward()
            return
        
        # 执行每个反向传播阶段
        for stage in self.backward_stages:
            stage.inputs = {"grad_output": loss} if stage.stage_id == 0 else {}
            stage.outputs = {}
            stage.execute()
            
            # 更新下一个阶段的输入
            if stage.stage_id < len(self.backward_stages) - 1:
                self.backward_stages[stage.stage_id + 1].inputs.update(stage.outputs)
        
        # 等待所有阶段完成
        torch.cuda.synchronize()


class BitPackEngine:
    """位打包执行引擎，处理脉冲信号的压缩存储"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.bit_pack_dim = self.config.bit_pack_dim
        
    def pack_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        将脉冲信号打包为位压缩格式
        Args:
            spikes: 浮点脉冲张量，形状为 [batch, ...]
        Returns:
            压缩后的整数张量
        """
        from .utils.memory import pack_spikes
        return pack_spikes(spikes, self.bit_pack_dim)
    
    def unpack_spikes(self, packed_spikes: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        将位压缩格式的信号解包为原始脉冲
        Args:
            packed_spikes: 压缩后的整数张量
            original_shape: 原始形状
        Returns:
            浮点脉冲张量
        """
        from .utils.memory import unpack_spikes
        return unpack_spikes(packed_spikes, original_shape, self.bit_pack_dim)