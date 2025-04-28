# snn_pipeline/examples/mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_pipeline import SpikingConv2d, SpikingModule, Config, Trainer
from snn_pipeline.layers import InputEncoder, OutputDecoder, FlattenLayer

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 配置参数
config = Config()
config.time_steps = 10
config.tau_mem = 5.0
config.threshold = 1.0
config.reset_mode = "zero"
config.surrogate_gradient = "fast_sigmoid"
config.surrogate_slope = 1000
config.pipeline_width = 4
config.bit_pack = True

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
class SpikingMNIST(SpikingModule):
    def __init__(self, time_steps=10, config=None):
        super().__init__(time_steps, config)
        
        # 输入编码器
        self.input_encoder = InputEncoder((1, 28, 28), "direct", time_steps)
        
        # 卷积层
        self.conv1 = SpikingConv2d(1, 32, kernel_size=3, stride=1, padding=1,
                                  neuron_model="LIF", tau_mem=config.tau_mem,
                                  threshold=config.threshold, reset_mode=config.reset_mode,
                                  surrogate_gradient=config.surrogate_gradient,
                                  surrogate_slope=config.surrogate_slope,
                                  bit_pack=config.bit_pack)
        
        self.conv2 = SpikingConv2d(32, 64, kernel_size=3, stride=2, padding=1,
                                  neuron_model="LIF", tau_mem=config.tau_mem,
                                  threshold=config.threshold, reset_mode=config.reset_mode,
                                  surrogate_gradient=config.surrogate_gradient,
                                  surrogate_slope=config.surrogate_slope,
                                  bit_pack=config.bit_pack)
        
        self.conv3 = SpikingConv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                  neuron_model="LIF", tau_mem=config.tau_mem,
                                  threshold=config.threshold, reset_mode=config.reset_mode,
                                  surrogate_gradient=config.surrogate_gradient,
                                  surrogate_slope=config.surrogate_slope,
                                  bit_pack=config.bit_pack)
        
        # 全连接层
        self.fc1 = nn.Linear(7*7*128,1024)
        
        self.fc2 = nn.Linear(1024,10)
        
        self.flatten = FlattenLayer()
        
        # 输出解码器
        self.output_decoder = OutputDecoder("rate")
        
        # 添加到层列表
        self.layers = nn.ModuleList([
            self.conv1, self.conv2, self.conv3, self.flatten,
            self.fc1, self.fc2
        ])
    
    def forward(self, x, time_steps=None):
        t_steps = time_steps or self.time_steps
        spike_inputs = self.input_encoder(x)
        self.reset_states()
        spike_outputs = []
        
        for t in range(t_steps):
            x_t = spike_inputs[t]
            x_t = self.conv1(x_t)
            x_t = self.conv2(x_t)
            x_t = self.conv3(x_t)
            
            x_t = self.flatten(x_t)   
            
            spike_outputs.append(x_t)
        
        x = self.output_decoder(spike_outputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpikingMNIST(config.time_steps, config).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义评估指标
def accuracy(y_true, y_pred):
    pred = y_pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(y_true.view_as(pred)).sum().float()
    return correct / y_true.size(0)

# 训练
trainer = Trainer(model, criterion, optimizer, config)
trainer.compile(metrics={'accuracy': accuracy}, use_amp=torch.cuda.is_available())

# 添加回调
from snn_pipeline.training import EarlyStopping, ModelCheckpoint
trainer.add_callback(EarlyStopping(monitor='val_loss', patience=5))
trainer.add_callback(ModelCheckpoint('mnist_snn_model.pt', monitor='val_accuracy', save_best_only=True))

# 训练
history = trainer.fit(train_loader, test_loader, epochs=10, verbose=1)

# 保存模型
torch.save(model.state_dict(), 'mnist_snn_final.pt')

print("Training completed!")


if __name__ == "__main__":
    # 运行示例
    print("Running MNIST SNN example...")
    print(f"Device: {device}")
    print(f"Time steps: {config.time_steps}")
    print(f"Pipeline width: {config.pipeline_width}")
    
    # 评估最终模型
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Final test accuracy: {100 * correct / total:.2f}%")