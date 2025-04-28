# snn_pipeline/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from tqdm import tqdm
import time
import logging

from .config import Config
from .engine import PipelineEngine

class Callback:
    """训练回调基类"""
    
    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None:
        pass


class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0, patience: int = 0, 
                 mode: str = 'auto', baseline: Optional[float] = None):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.stop_training = False
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b + min_delta
        else:  # auto
            if 'acc' in monitor:
                self.monitor_op = lambda a, b: a > b + min_delta
            else:
                self.monitor_op = lambda a, b: a < b - min_delta
    
    def on_train_begin(self, logs: Dict[str, Any] = None) -> None:
        self.wait = 0
        self.best = float('inf') if self.monitor_op(0, 1) else float('-inf')
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.baseline is not None and epoch == 0:
            self.best = self.baseline
            
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True


class ModelCheckpoint(Callback):
    """模型检查点回调"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = False, 
                 mode: str = 'auto', save_freq: Union[str, int] = 'epoch'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.mode = mode
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b
            self.best = float('-inf')
        else:  # auto
            if 'acc' in monitor:
                self.monitor_op = lambda a, b: a > b
                self.best = float('-inf')
            else:
                self.monitor_op = lambda a, b: a < b
                self.best = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        if self.save_freq == 'epoch':
            self._save_model(epoch, logs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        if isinstance(self.save_freq, int) and (batch + 1) % self.save_freq == 0:
            self._save_model(batch, logs)
    
    def _save_model(self, index: int, logs: Dict[str, Any]) -> None:
        filepath = self.filepath.format(epoch=index, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                return
                
            if self.monitor_op(current, self.best):
                self.best = current
                if hasattr(self, 'model'):
                    torch.save(self.model.state_dict(), filepath)
        else:
            if hasattr(self, 'model'):
                torch.save(self.model.state_dict(), filepath)


class LearningRateScheduler(Callback):
    """学习率调度器回调"""
    
    def __init__(self, schedule: Callable[[int], float], verbose: int = 0):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None) -> None:
        if not hasattr(self, 'optimizer'):
            return
            
        lr = self.schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        if self.verbose > 0:
            print(f'Epoch {epoch}: LearningRateScheduler setting learning rate to {lr}.')


class Trainer:
    """脉冲神经网络训练器"""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, 
                 config: Optional[Config] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config or Config()
        self.engine = PipelineEngine(config)
        self.callbacks = []
        self.stop_training = False
        self.metrics = {}
        self.use_amp = False  # 自动混合精度
        self.scaler = None
        
    def compile(self, metrics: Dict[str, Callable] = None, use_amp: bool = False) -> None:
        """
        编译训练器
        Args:
            metrics: 评估指标
            use_amp: 是否使用自动混合精度
        """
        self.metrics = metrics or {}
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def add_callback(self, callback: Callback) -> None:
        """添加回调函数"""
        self.callbacks.append(callback)
        # 将trainer信息传递给回调
        callback.model = self.model
        callback.optimizer = self.optimizer
    
    def _call_callbacks(self, method: str, *args, **kwargs) -> None:
        """调用回调函数"""
        for callback in self.callbacks:
            if hasattr(callback, method):
                method_fn = getattr(callback, method)
                method_fn(*args, **kwargs)
                
                # 检查是否需要停止训练
                if hasattr(callback, 'stop_training') and callback.stop_training:
                    self.stop_training = True
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        单步训练
        Args:
            x: 输入数据
            y: 目标标签
        Returns:
            损失和指标值
        """
        # 构建流水线（如果尚未构建）
        if not hasattr(self.engine, 'is_built') or not self.engine.is_built:
            self.engine.build_pipeline(self.model, x)
        
        # 训练模式
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播（使用流水线）
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.engine.forward(x, self.config.time_steps)
                # print(outputs.requires_grad, outputs.grad_fn)     # 预期: True, 不为 None
                loss = self.criterion(outputs, y)
                # print(loss.grad_fn)                       # 也应当不为 None
        else:
            outputs = self.engine.forward(x, self.config.time_steps)
            # print(outputs.requires_grad, outputs.grad_fn)     # 预期: True, 不为 None
            loss = self.criterion(outputs, y)
            # print(loss.grad_fn)                       # 也应当不为 None
        
        # 反向传播
        # self.optimizer.zero_grad()
        # if self.use_amp:
        #     self.scaler.scale(loss)           # 先 scale
        #     self.engine.backward(loss)        # 手动流水线反向
        #     self.scaler.step(self.optimizer)
        #     self.scaler.update()
        # else:
        #     self.engine.backward(loss)
        #     self.optimizer.step()

        # 使用 PyTorch 的自动微分

        # 反向传播
        loss.backward()

        # 调试打印：一定要在 loss.backward() 之后！
        for name, p in self.model.named_parameters():
            if p.grad is None:
                print(f"[WARN] {name}.grad is None")
            else:
                print(f"{name}.grad norm = {p.grad.norm().item():.6f}")
            
        self.optimizer.step()
        
        # 计算指标
        result = {'loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            result[name] = metric_fn(y, outputs).item()
            
        return result
    
    def val_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        单步验证
        Args:
            x: 输入数据
            y: 目标标签
        Returns:
            损失和指标值
        """
        # 构建流水线（如果尚未构建）
        if not hasattr(self.engine, 'is_built') or not self.engine.is_built:
            self.engine.build_pipeline(self.model, x)
        
        # 评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播（使用流水线）
            outputs = self.engine.forward(x, self.config.time_steps)
            loss = self.criterion(outputs, y)
        
        # 计算指标
        result = {'val_loss': loss.item()}
        for name, metric_fn in self.metrics.items():
            result[f'val_{name}'] = metric_fn(y, outputs).item()
            
        return result
    
    def fit(self, train_loader: torch.utils.data.DataLoader, 
            val_loader: Optional[torch.utils.data.DataLoader] = None, 
            epochs: int = 10, verbose: int = 1) -> Dict[str, List[float]]:
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            verbose: 日志详细程度
        Returns:
            训练历史
        """
        # 训练历史
        history = {
            'loss': [], 
            'val_loss': []
        }
        for name in self.metrics:
            history[name] = []
            history[f'val_{name}'] = []
        
        # 回调：训练开始
        logs = {}
        self._call_callbacks('on_train_begin', logs)
        
        # 训练循环
        for epoch in range(epochs):
            if self.stop_training:
                break
                
            # 回调：轮次开始
            epoch_logs = {}
            self._call_callbacks('on_epoch_begin', epoch, epoch_logs)
            
            # 训练
            if verbose > 0:
                print(f'Epoch {epoch+1}/{epochs}')
                train_pbar = tqdm(train_loader)
            else:
                train_pbar = train_loader
                
            epoch_metrics = {key: 0.0 for key in ['loss'] + list(self.metrics.keys())}
            batch_count = 0
            
            for batch_idx, (x, y) in enumerate(train_pbar):
                if self.stop_training:
                    break
                    
                # 回调：批次开始
                batch_logs = {}
                self._call_callbacks('on_batch_begin', batch_idx, batch_logs)
                
                # 移动数据到设备
                x = x.to(next(self.model.parameters()).device)
                y = y.to(next(self.model.parameters()).device)
                
                # 训练步骤
                batch_result = self.train_step(x, y)
                
                # 更新epoch指标累计
                for key, value in batch_result.items():
                    epoch_metrics[key] += value
                batch_count += 1
                
                # 更新进度条
                if verbose > 0:
                    desc = ' - '.join([f'{k}: {v:.4f}' for k, v in batch_result.items()])
                    train_pbar.set_description(desc)
                
                # 回调：批次结束
                batch_logs.update(batch_result)
                self._call_callbacks('on_batch_end', batch_idx, batch_logs)
            
            # 计算epoch平均指标
            for key in epoch_metrics:
                epoch_metrics[key] /= batch_count
                history[key].append(epoch_metrics[key])
            
            # 验证
            if val_loader is not None:
                if verbose > 0:
                    val_pbar = tqdm(val_loader, desc='Validation')
                else:
                    val_pbar = val_loader
                    
                val_metrics = {key: 0.0 for key in ['val_loss'] + [f'val_{name}' for name in self.metrics]}
                val_count = 0
                
                for x, y in val_pbar:
                    # 移动数据到设备
                    x = x.to(next(self.model.parameters()).device)
                    y = y.to(next(self.model.parameters()).device)
                    
                    # 验证步骤
                    val_result = self.val_step(x, y)
                    
                    # 更新验证指标累计
                    for key, value in val_result.items():
                        val_metrics[key] += value
                    val_count += 1
                
                # 计算验证平均指标
                for key in val_metrics:
                    val_metrics[key] /= val_count
                    history[key].append(val_metrics[key])
                
                # 更新epoch日志
                epoch_logs.update(val_metrics)
            
            # 打印epoch结果
            if verbose > 0:
                # 构建日志消息
                msg_parts = []
                for key, value in {**epoch_metrics, **(val_metrics if val_loader else {})}.items():
                    msg_parts.append(f'{key}: {value:.4f}')
                print(' - '.join(msg_parts))
            
            # 回调：轮次结束
            epoch_logs.update(epoch_metrics)
            self._call_callbacks('on_epoch_end', epoch, epoch_logs)
        
        # 回调：训练结束
        self._call_callbacks('on_train_end', logs)
        
        return history