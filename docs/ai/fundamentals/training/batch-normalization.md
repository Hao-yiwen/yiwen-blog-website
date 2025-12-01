---
title: Batch Normalization 批归一化
sidebar_label: Batch Normalization
date: 2025-01-12
last_update:
  date: 2025-01-12
---

# Batch Normalization 批归一化

## 什么是 Batch Normalization？

Batch Normalization (BN) 是一种在深度神经网络训练过程中**持续稳定激活值分布**的技术。它是网络结构的一部分，在每次前向传播时都会执行。

## 为什么需要 BN？

在深度网络训练过程中，由于参数不断更新，每层的输入分布会持续变化（Internal Covariate Shift），导致：

- **训练不稳定**：每层都需要不断适应新的输入分布
- **收敛速度慢**：需要使用较小的学习率
- **梯度问题**：容易出现梯度消失或爆炸

BN 通过在每个 mini-batch 上归一化激活值，稳定输入分布。

## 工作原理

对于一个 mini-batch 的数据 $\mathcal{B} = \{x_1, x_2, ..., x_m\}$：

**1. 计算均值和方差**
$$
\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}x_i, \quad \sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2
$$

**2. 归一化**
$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

**3. 缩放和平移（可学习参数）**
$$
y_i = \gamma \hat{x}_i + \beta
$$

其中 $\gamma$ 和 $\beta$ 是可学习参数，让网络能够学习最优的分布。

## PyTorch 实现

### 标准使用方式

**推荐顺序：Conv -> BN -> Activation**

```python
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 BN 时可以省略 conv 的 bias
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        # Conv -> BN -> Activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
```

### 训练和推理的区别

```python
# 训练模式：使用当前 batch 的统计量
model.train()

# 推理模式：使用训练时累积的移动平均统计量
model.eval()
```

### 不同类型的 BN

| BN 类型 | 适用场景 |
|--------|---------|
| `nn.BatchNorm1d(features)` | 全连接层 |
| `nn.BatchNorm2d(channels)` | 2D 卷积层（最常用）|
| `nn.BatchNorm3d(channels)` | 3D 卷积层 |

## BN 的优势

1. **允许更大的学习率**：激活值分布稳定，梯度尺度更一致
2. **降低对初始化的敏感性**：但仍需要好的权重初始化
3. **正则化效果**：mini-batch 噪声类似 Dropout，可减少 Dropout 使用
4. **加速收敛**：显著减少训练所需的 epoch 数量

## 使用注意事项

### Batch Size 影响

BN 效果**依赖于 batch size**：
- **推荐**：batch size >= 16
- **小 batch 替代方案**：Group Normalization (GN) 或 Layer Normalization (LN)

### 卷积层的 bias 可省略

由于 BN 有 $\beta$ 参数，卷积层的 bias 会被抵消：

```python
# bias=False 减少参数量
self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
self.bn = nn.BatchNorm2d(out_channels)
```

## 完整示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 8 * 8, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # γ = 1
                nn.init.constant_(m.bias, 0)     # β = 0

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
```

## BN 的变体对比

| 归一化方法 | Batch 依赖 | 主要应用 |
|-----------|-----------|---------|
| Batch Norm | ✓ | CNN (大 batch) |
| Layer Norm | ✗ | Transformer, RNN |
| Instance Norm | ✗ | 风格迁移 |
| Group Norm | ✗ | CNN (小 batch) |

## 参考资料

- [原始论文: Batch Normalization (2015)](https://arxiv.org/abs/1502.03167)
- [动手学深度学习 - 批量归一化](https://zh-v2.d2l.ai/chapter_convolutional-modern/batch-norm.html)
