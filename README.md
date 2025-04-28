
初学者不懂乱写的项目，速度远慢于pytorch原版实现。

snn_pipeline/
├── snn_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   └── misc.py
│   ├── kernels/
│   │   ├── spike_kernels.cu          # CUDA kernel 实现
│   │   └── spike_kernels.py          # PyTorch CUDA binding
│   ├── engine.py                     # 核心流水线调度 & 执行器
│   ├── graph.py                      # 计算图拆分与重组
│   ├── layers.py                     # 高层 spiking 神经网络层（Conv, Linear…）
│   ├── neurons.py                    # 各类尖峰神经元模型（LIF, Izhikevich…）
│   ├── network.py                    # Model 定义 API（类似 torch.nn.Module）
│   ├── optim.py                      # 优化器扩展（AdamW, SGD…）
│   ├── training.py                   # 训练循环封装（forward+pipelines+backward）
│   ├── data.py                       # 数据加载与时序数据增强
│   └── examples/
│       └── mnist.py                  # 示例：如何定义模型、训练、评估
├── tests/
│   ├── test_memory.py
│   ├── test_engine.py
│   └── …
├── setup.py
└── README.md
```

以下是各文件/模块的职责说明：

1. snn_pipeline/config.py  
   - 全局超参数与默认配置（如 time‐step T、batch 大小、memory packing 策略等）  
   - 支持从 YAML/JSON/env 读取自定义配置，但还没做  

2. snn_pipeline/utils/memory.py  
   - 位打包/解包函数（bit‐packed spikes）  
   - 内存对齐、显存复用策略  
   - Tensor 缓存与回收池  

3. snn_pipeline/utils/misc.py  
   - 公共工具函数（时间戳、日志、seed 初始化等） 
   没写 

4. snn_pipeline/kernels/spike_kernels.cu  
   - 自定义 CUDA kernel：  
     • 基于位运算的脉冲传播  
     • 脉冲累积 / 重置  
     • 支持流水线化的前向算子  
   实际上还没用到。

5. snn_pipeline/kernels/spike_kernels.py  
   - PyTorch 自定义运算符封装（`torch.autograd.Function` + `load_cuda_extension`）  
   - 定义 forward/backward interface  

6. snn_pipeline/engine.py  
   - 构造“指令＋流水线”执行单元（类似 CPU pipeline stage）  
   - 接受 layers 序列与 time‐steps，将 O(T·L) 展开为可并行的 pipeline tasks  
   - 管理 kernel 调度、stream、event  

7. snn_pipeline/graph.py  
   - 前向阶段：生成紧耦合流水线任务图  
   - 反向阶段：按原始串行顺序重建计算图，以支持标准反向传播  
   - GraphBlock/Node 数据结构  

8. snn_pipeline/layers.py  
   - 定义高层神经网络层：`SpikingConv2d`, `SpikingLinear`, `SpikingBatchNorm` 等  
   - 每个 layer“包装”相应 neuron + kernel 调用  
   - 支持 hyperparameters（theta, tau, threshold…）  

9. snn_pipeline/neurons.py  
   - 各种 neuron 动态模型（LIF, ERF, Izhikevich…）  
   - surrogate gradient 函数定义  
   - 状态更新（membrane potential, spike generation）  

10. snn_pipeline/network.py  
    - `class SpikingModule(torch.nn.Module)` 基类  
    - 支持`add_layer()`、序列／图结构定义  
    - 提供`forward(inputs, time_steps)` 简洁接口  

11. snn_pipeline/optim.py  
    - 在脉冲网络上可能的特殊优化器（带有时序权重衰减等）  
    - 继承自`torch.optim.Optimizer`
    还没写  

12. snn_pipeline/training.py  
    - 封装训练流程：  
      • 数据加载 → forward（engine 调用） → loss → backward → optimizer.step()  
      • 支持梯度累积、混合精度、显存监控  
    - Callback 机制（学习率调度、早停、日志）  

13. snn_pipeline/data.py  
    - 时序数据专用 DataLoader/Dataset（如 neuromorphic event 数据集）  
    - 预处理（time‐binned conversion, normalization 等）  

14. snn_pipeline/examples/  
    - 完整示例脚本：如何定义网络、设置 config、启动训练、评估  

15. tests/  
    - 单元测试：memory packing、kernel 输出正确性、engine pipeline 准确性  
   没写

16. setup.py & README.md  
    - 项目安装、依赖说明、快速上手指南  

