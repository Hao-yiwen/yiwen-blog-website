---
title: LeNet
sidebar_label: LeNet
date: 2025-01-17
last_update:
  date: 2025-01-17
---

# LeNet - 开创性的卷积神经网络

## 历史背景

LeNet 是最早的卷积神经网络之一，由 Yann LeCun 在 1989 年提出，主要用于手写数字识别（MNIST 数据集）。在深度学习复兴之前，LeNet 就已经在商业应用中取得了成功，例如在 ATM 机上识别支票数字。

虽然现在看来它的结构相对简单，但 LeNet 奠定了现代卷积神经网络的基础架构，是深度学习历史上的里程碑。

## 网络结构

LeNet-5 由两部分组成：**卷积编码器**和**全连接层**

### 完整架构

```
输入 (1×28×28)
    ↓
Conv1 (6个5×5卷积核) → Sigmoid → AvgPool (2×2)
    ↓ (6×14×14)
Conv2 (16个5×5卷积核) → Sigmoid → AvgPool (2×2)
    ↓ (16×5×5)
Flatten → FC1 (120) → Sigmoid
    ↓
FC2 (84) → Sigmoid
    ↓
FC3 (10个输出)
```

### 层级详解

1. **第一个卷积块**
   - 输入：1 通道（灰度图像）28×28
   - 卷积层：6 个 5×5 卷积核，输出 6×28×28
   - 激活函数：Sigmoid
   - 平均池化：2×2，输出 6×14×14

2. **第二个卷积块**
   - 卷积层：16 个 5×5 卷积核，输出 16×10×10
   - 激活函数：Sigmoid
   - 平均池化：2×2，输出 16×5×5

3. **全连接层**
   - 展平：16×5×5 = 400
   - FC1：400 → 120（Sigmoid）
   - FC2：120 → 84（Sigmoid）
   - FC3：84 → 10（输出层，10个数字类别）

## PyTorch 实现

```python
import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积编码器
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积块 1
        x = self.sigmoid(self.conv1(x))
        x = self.avgpool(x)

        # 卷积块 2
        x = self.sigmoid(self.conv2(x))
        x = self.avgpool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x

# 创建模型
net = LeNet()
print(net)
```

### 查看每层输出形状

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

输出：
```
Conv2d output shape:     torch.Size([1, 6, 28, 28])
Sigmoid output shape:    torch.Size([1, 6, 28, 28])
AvgPool2d output shape:  torch.Size([1, 6, 14, 14])
Conv2d output shape:     torch.Size([1, 16, 10, 10])
Sigmoid output shape:    torch.Size([1, 16, 10, 10])
AvgPool2d output shape:  torch.Size([1, 16, 5, 5])
Flatten output shape:    torch.Size([1, 400])
Linear output shape:     torch.Size([1, 120])
Sigmoid output shape:    torch.Size([1, 120])
Linear output shape:     torch.Size([1, 84])
Sigmoid output shape:    torch.Size([1, 84])
Linear output shape:     torch.Size([1, 10])
```

## 核心设计思想

### 1. 保留空间结构

与传统的多层感知机不同，LeNet 使用卷积层而非全连接层处理图像，避免了将图像展平导致的空间信息丢失。

### 2. 参数共享

卷积核在整个图像上滑动，大大减少了参数数量。相比全连接层，LeNet 更加轻量高效。

### 3. 层次化特征提取

- **浅层**：提取简单特征（边缘、轮廓）
- **深层**：组合成复杂特征（数字形状）
- **全连接层**：进行最终分类

### 4. 逐层缩小空间维度，增加通道数

- 输入：1×28×28
- Conv1 后：6×14×14
- Conv2 后：16×5×5

这种设计模式成为后续 CNN 架构的标准范式。

## 现代改进

虽然 LeNet 在当时非常成功，但现代 CNN 通常会做以下改进：

- **激活函数**：Sigmoid → **ReLU**（解决梯度消失）
- **池化方式**：平均池化 → **最大池化**（保留显著特征）
- **正则化**：增加 **Dropout**、**Batch Normalization**
- **优化器**：使用 **Adam** 而非传统的 SGD

## 训练示例

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据准备
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

# 模型、损失、优化器
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
model.train()
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 性能

在 MNIST 数据集上，LeNet 可以达到约 **98-99%** 的准确率，这在当时是突破性的成果。

## 总结

LeNet 的贡献不仅在于其性能，更在于：

1. **证明了 CNN 的有效性**：在图像识别任务上超越传统方法
2. **确立了 CNN 架构范式**：卷积层 + 池化层 + 全连接层
3. **启发了后续研究**：AlexNet、VGG、ResNet 等都沿用了类似设计

虽然今天我们有更强大的模型，但 LeNet 仍然是学习卷积神经网络的最佳起点。

## 参考资料

- 原始论文：[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- 动手学深度学习：[LeNet 章节](https://zh-v2.d2l.ai/chapter_convolutional-neural-networks/lenet.html)
