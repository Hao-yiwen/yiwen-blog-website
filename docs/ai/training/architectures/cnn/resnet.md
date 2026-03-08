# ResNet 残差网络详解

ResNet(Residual Network,残差网络)是2015年由微软研究院的何恺明等人提出的深度卷积神经网络架构,在ImageNet竞赛中取得了突破性成果。

## 核心问题

在ResNet之前,研究者发现一个矛盾现象:**网络越深,训练效果反而越差**。这不是过拟合,而是训练集上的错误率都会上升,称为"退化问题"(degradation problem)。

## 核心创新:残差块

ResNet提出了**残差连接**(skip connection/shortcut connection)。基本残差块的PyTorch实现:

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet基本残差块(用于ResNet-18/34)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出维度不同,需要调整shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x  # 保存输入

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接:F(x) + x
        out += self.shortcut(identity)
        out = self.relu(out)

        return out
```

**关键洞察**:让网络学习残差 `F(x) = H(x) - x` 比直接学习目标映射 `H(x)` 更容易。如果恒等映射是最优的,网络只需把 `F(x)` 的权重推向零即可。

## 主要变体

- **ResNet-18, 34**: 使用BasicBlock,层数较少
- **ResNet-50, 101, 152**: 使用Bottleneck,参数更高效

## 重要影响

1. **使极深网络成为可能**:可以轻松训练上百层甚至上千层的网络
2. **梯度流动更顺畅**:跳跃连接为梯度提供了"高速公路",缓解梯度消失
3. **后续架构的基础**:ResNet的思想影响了几乎所有后续的深度学习架构(Transformer、ViT等)

ResNet是深度学习历史上的里程碑,证明了"深度"确实能带来更强的表达能力,前提是有合适的架构设计。
