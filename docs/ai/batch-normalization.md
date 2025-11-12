---
title: Batch Normalization 批归一化
sidebar_label: Batch Normalization
date: 2025-01-12
last_update:
  date: 2025-01-12
---

# Batch Normalization 批归一化

## 什么是 Batch Normalization？

Batch Normalization (BN) 是一种在深度神经网络训练过程中**持续稳定激活值分布**的技术，由 Sergey Ioffe 和 Christian Szegedy 在 2015 年提出。它是网络结构的一部分，在每次前向传播时都会执行。

## 为什么需要 Batch Normalization？

### 问题：内部协变量偏移（Internal Covariate Shift）

在深度网络训练过程中，由于参数不断更新，每层的输入分布会持续变化，这会导致：

1. **训练不稳定**：每层都需要不断适应新的输入分布
2. **收敛速度慢**：需要使用较小的学习率来保证稳定性
3. **梯度问题**：容易出现梯度消失或爆炸

### 解决方案：归一化激活值

BN 通过在每个 mini-batch 上归一化激活值，使得每层的输入分布保持相对稳定。

## Batch Normalization 的工作原理

### 数学公式

对于一个 mini-batch 的数据 $\mathcal{B} = \{x_1, x_2, ..., x_m\}$：

**1. 计算均值和方差**
$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2
$$

**2. 归一化**
$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

其中 $\epsilon$ 是一个很小的常数（如 $10^{-5}$），防止除以零。

**3. 缩放和平移（可学习参数）**
$$
y_i = \gamma \hat{x}_i + \beta
$$

其中：
- $\gamma$ (scale)：缩放参数，可学习
- $\beta$ (shift)：平移参数，可学习

### 为什么需要 γ 和 β？

归一化后的数据均值为 0，方差为 1，但这可能**限制了网络的表达能力**。通过引入可学习的 $\gamma$ 和 $\beta$，网络可以：
- 恢复原始分布（如果需要）：$\gamma = \sqrt{\sigma_\mathcal{B}^2}$, $\beta = \mu_\mathcal{B}$
- 学习最优的分布参数

## PyTorch 实现

### 基本使用

```python
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # 对 64 个通道进行 BN

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        # 标准流程：Conv -> BN -> Activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
```

### BN 的位置选择

有两种常见的放置方式：

**方式 1：Conv -> BN -> Activation（推荐）**
```python
x = self.conv(x)
x = self.bn(x)
x = F.relu(x)
```

**方式 2：Conv -> Activation -> BN**
```python
x = self.conv(x)
x = F.relu(x)
x = self.bn(x)
```

原论文推荐方式 1，因为在激活函数之前归一化可以防止 ReLU 进入饱和区。

### 不同类型的 BN

| BN 类型 | 适用场景 | 归一化维度 |
|--------|---------|----------|
| `nn.BatchNorm1d(features)` | 全连接层 | (N, C) 或 (N, C, L) |
| `nn.BatchNorm2d(channels)` | 2D 卷积层 | (N, C, H, W) |
| `nn.BatchNorm3d(channels)` | 3D 卷积层 | (N, C, D, H, W) |

### 训练和推理的区别

```python
# 训练模式
model.train()
# BN 使用当前 batch 的均值和方差

# 推理模式
model.eval()
# BN 使用训练时累积的移动平均均值和方差
```

训练时，BN 会维护一个**移动平均**的均值和方差（momentum 参数控制）：

```python
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var = (1 - momentum) * running_var + momentum * batch_var
```

## Batch Normalization 的优势

### 1. 允许更大的学习率
- 激活值分布稳定，梯度尺度更一致
- 可以使用更大的学习率，加速训练

### 2. 降低对初始化的敏感性
- 即使初始化不够理想，BN 也能帮助稳定训练
- 但**不能完全替代**好的权重初始化

### 3. 正则化效果
- 每个 mini-batch 的归一化引入了噪声
- 类似于 Dropout 的正则化效果，有助于防止过拟合
- 使用 BN 后可以减少或不使用 Dropout

### 4. 加速收敛
- 显著减少训练所需的 epoch 数量
- 在许多任务上能提升最终性能

## 使用注意事项

### 1. Batch Size 的影响

BN 的效果**依赖于 batch size**：
- **小 batch**：统计量估计不准确，可能导致性能下降
- **推荐**：batch size >= 16（越大越好，但受限于显存）

对于小 batch 场景，可以考虑：
- **Group Normalization (GN)**：不依赖 batch size
- **Layer Normalization (LN)**：常用于 Transformer

### 2. 与 Dropout 的配合

BN 本身有正则化效果，使用时注意：
- 使用 BN 后，通常可以减小 Dropout 概率
- 某些情况下可以完全移除 Dropout

### 3. 卷积层的 bias 可以省略

由于 BN 有 $\beta$ 参数（可学习的偏置），卷积层的 bias 会被抵消：

```python
# 使用 BN 时，可以不使用 conv bias
self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
self.bn = nn.BatchNorm2d(out_channels)
```

这样可以减少参数数量。

## Batch Normalization 的变体

### Group Normalization (GN)
- 将通道分组进行归一化
- 不依赖 batch size，适合小 batch 训练

### Layer Normalization (LN)
- 对每个样本的所有特征进行归一化
- 常用于 RNN 和 Transformer

### Instance Normalization (IN)
- 对每个样本的每个通道单独归一化
- 常用于风格迁移任务

### 对比表格

| 归一化方法 | 归一化维度 | Batch 依赖 | 主要应用 |
|-----------|----------|-----------|---------|
| Batch Norm | (N, H, W) | ✓ | CNN (大 batch) |
| Layer Norm | (C, H, W) | ✗ | Transformer, RNN |
| Instance Norm | (H, W) | ✗ | 风格迁移 |
| Group Norm | (C/G, H, W) | ✗ | CNN (小 batch) |

## 完整示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Conv Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Fully Connected
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

        # 初始化（即使有 BN 也推荐）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # γ 初始化为 1
                nn.init.constant_(m.bias, 0)     # β 初始化为 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 使用示例
model = ModernCNN(num_classes=10)

# 训练模式
model.train()
output = model(torch.randn(32, 3, 32, 32))  # batch_size=32

# 推理模式
model.eval()
with torch.no_grad():
    output = model(torch.randn(1, 3, 32, 32))  # 单张图片推理
```

## 参考资料

- [原始论文: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [动手学深度学习 - 批量归一化](https://zh-v2.d2l.ai/chapter_convolutional-modern/batch-norm.html)
