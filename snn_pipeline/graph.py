# snn_pipeline/graph.py
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union, Callable


class GraphNode:
    """计算图节点，表示计算图中的单个操作"""
    
    def __init__(self, op_type: str, op_func: Callable, inputs: List[str], outputs: List[str]):
        self.op_type = op_type
        self.op_func = op_func
        self.input_keys = inputs
        self.output_keys = outputs
        self.backward_node = None
        
    def execute(self, inputs: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> None:
        """执行操作"""
        # 1) 打印消费的 inputs
        # info_in = ", ".join(f"{k}:{tuple(inputs[k].shape)}" for k in self.input_keys)
        # print(f"[Node {self.op_type:^12} ▸ consume] {info_in}")
        # 2) 真正执行
        # 收集输入
        op_inputs = []
        for key in self.input_keys:
            if key not in inputs:
                raise KeyError(f"Input '{key}' not found for operation '{self.op_type}'")
            op_inputs.append(inputs[key])
        
        # 执行操作
        try:
            op_results = self.op_func(*op_inputs)
        except TypeError:
            op_results = self.op_func(op_inputs)
        
        # 如果结果不是元组或列表，将其转换为列表
        if not isinstance(op_results, (tuple, list)):
            op_results = [op_results]
        # 3) 打印产出的 outputs
        # info_out = ", ".join(f"{k}:{tuple(r.shape)}" for k, r in zip(self.output_keys, op_results))
        # print(f"[Node {self.op_type:^12} ◂ produce] {info_out}")
        # 4) 写回 outputs
        # 保存输出
        for key, result in zip(self.output_keys, op_results):
            outputs[key] = result


class GraphBlock:
    """计算图块，表示一组可以一起执行的操作"""
    
    def __init__(self, block_id: int, nodes: List[GraphNode] = None):
        self.block_id = block_id
        self.nodes = nodes or []
        self.inputs = set()
        self.outputs = set()
        self.dependencies = set()  # 依赖的块ID
        self.dependents = set()    # 依赖于本块的块ID
        
    def add_node(self, node: GraphNode) -> None:
        """添加节点"""
        self.nodes.append(node)
        # 更新输入输出集合
        self.inputs.update(node.input_keys)
        self.outputs.update(node.output_keys)
        
    def can_merge(self, other: 'GraphBlock') -> bool:
        """检查是否可以合并两个块"""
        # 如果两个块之间有依赖，不能合并
        if self.block_id in other.dependencies or other.block_id in self.dependencies:
            return False
        
        # 如果合并后输入大小超过阈值，不合并
        merged_inputs = self.inputs.union(other.inputs)
        if len(merged_inputs) > 20:  # 阈值可调
            return False
            
        return True
    
    def merge(self, other: 'GraphBlock') -> 'GraphBlock':
        """合并两个块"""
        # 创建新块
        new_block = GraphBlock(self.block_id)
        new_block.nodes = self.nodes + other.nodes
        new_block.inputs = self.inputs.union(other.inputs)
        new_block.outputs = self.outputs.union(other.outputs)
        new_block.dependencies = self.dependencies.union(other.dependencies)
        new_block.dependents = self.dependents.union(other.dependents)
        
        # 移除自引用
        if new_block.block_id in new_block.dependencies:
            new_block.dependencies.remove(new_block.block_id)
        if new_block.block_id in new_block.dependents:
            new_block.dependents.remove(new_block.block_id)
            
        return new_block


class ComputationGraph:
    """计算图，用于管理操作调度"""
    
    def __init__(self):
        self.forward_blocks = []  # 前向计算块
        self.backward_nodes = []  # 反向计算节点
        self.input_shapes = {}    # 输入形状
        
    def build_forward_graph(self, model: nn.Module, sample_input: torch.Tensor, time_steps: int) -> None:
        """
        构建前向计算图：
          - 将 model.layers 在每个时间步 t 展开成一个 GraphBlock
          - 每个 block 中包含 L = len(model.layers) 个 GraphNode
          - 如果有 output_decoder，则在所有时间步之外再加一个解码器 block
          - 建立 block 之间的依赖关系（线性串行 + decoder 依赖所有 time‐step blocks）
          - 调用 _optimize_forward_graph 合并无依赖的相邻 blocks
        """
        # 清空旧图
        self.forward_blocks = []
        # 记录一下输入形状
        self.input_shapes['input'] = tuple(sample_input.shape)
        
        # 简化起见，不做实际前向计算，只根据 model.layers 顺序构建节点
        # 第 0..T-1 个 block 对应时序中的每一步
        for t in range(time_steps):
            blk = GraphBlock(block_id=t)
            prev_key = 'input'
            
            # 对应每一层生成一个 GraphNode
            for idx, layer in enumerate(model.layers):
                op_type = layer.__class__.__name__
                # 直接绑定到 layer.forward，后续真实执行时会调用该方法
                op_func = layer.forward  
                
                # 前一个节点的输出 key
                input_keys = [prev_key]
                # 为本节点生成一个唯一输出 key
                output_key = f"{op_type}_t{t}_{idx}"
                
                node = GraphNode(op_type=op_type,
                                 op_func=op_func,
                                 inputs=input_keys,
                                 outputs=[output_key])
                blk.add_node(node)
                
                prev_key = output_key
            
            self.forward_blocks.append(blk)
        
        # 如果有解码器，则在 time_steps 之外再加一个 block
        if hasattr(model, 'output_decoder') and model.output_decoder is not None:
            decoder = model.output_decoder
            blk = GraphBlock(block_id=time_steps)
            op_type = decoder.__class__.__name__
            op_func = decoder.forward
            
            # 输入是所有 t 最后一层的输出 key
            last_layer = model.layers[-1]
            last_op = last_layer.__class__.__name__
            input_keys = [
                f"{last_op}_t{t}_{len(model.layers)-1}"
                for t in range(time_steps)
            ]
            output_key = 'output'
            
            node = GraphNode(op_type=op_type,
                             op_func=op_func,
                             inputs=input_keys,
                             outputs=[output_key])
            blk.add_node(node)
            self.forward_blocks.append(blk)
        
        # 构建各 block 之间的依赖（线性链）
        N = len(self.forward_blocks)
        for i in range(N):
            blk = self.forward_blocks[i]
            # 普通时序块依赖前一个时序块
            if i > 0 and i < time_steps:
                blk.dependencies.add(i-1)
            # 解码器 block（i == time_steps）依赖所有时序块
            if i == time_steps:
                blk.dependencies.update(range(time_steps))
        
        # 同时填充 dependents
        for i, blk in enumerate(self.forward_blocks):
            for dep in blk.dependencies:
                self.forward_blocks[dep].dependents.add(i)
        
        # 最后做一次前向图合并优化
        self._optimize_forward_graph()

    def build_backward_graph(self) -> None:
        """构建反向计算图"""
        # 反向图按照正常的反向顺序构建，不进行流水线优化
        self.backward_nodes = []
        
        # 为每个前向块创建对应的反向节点
        for block in reversed(self.forward_blocks):
            for node in reversed(block.nodes):
                # 简化示例：为每个前向节点创建一个反向节点
                backward_node = GraphNode(
                    op_type=f"{node.op_type}_backward",
                    op_func=lambda grad: grad,  # 简化的反向传播函数
                    inputs=[f"grad_{key}" for key in node.output_keys],
                    outputs=[f"grad_{key}" for key in node.input_keys]
                )
                
                # 记录反向节点
                node.backward_node = backward_node
                self.backward_nodes.append(backward_node)
    
    def _optimize_forward_graph(self) -> None:
        """优化前向计算图，合并可并行执行的块"""
        # 简化示例：合并没有相互依赖的相邻块
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.forward_blocks) - 1:
                block1 = self.forward_blocks[i]
                block2 = self.forward_blocks[i + 1]
                
                if block1.can_merge(block2):
                    merged_block = block1.merge(block2)
                    self.forward_blocks[i] = merged_block
                    self.forward_blocks.pop(i + 1)
                    changed = True
                else:
                    i += 1
        
        # 重新分配块ID
        for i, block in enumerate(self.forward_blocks):
            block.block_id = i
            
        # 更新依赖关系
        for block in self.forward_blocks:
            new_deps = set()
            for dep_id in block.dependencies:
                for new_block in self.forward_blocks:
                    for node in new_block.nodes:
                        if node in self.forward_blocks[dep_id].nodes:
                            new_deps.add(new_block.block_id)
                            break
            block.dependencies = new_deps
            
            new_deps = set()
            for dep_id in block.dependents:
                for new_block in self.forward_blocks:
                    for node in new_block.nodes:
                        if node in self.forward_blocks[dep_id].nodes:
                            new_deps.add(new_block.block_id)
                            break
            block.dependents = new_deps