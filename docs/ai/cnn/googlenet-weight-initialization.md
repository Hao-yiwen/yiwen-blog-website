---
title: GoogleNet 权重初始化踩坑记录
sidebar_label: GoogleNet 权重初始化
date: 2025-01-12
last_update:
  date: 2025-01-12
---

# GoogleNet 训练问题复盘：权重初始化踩坑记录

## 问题现象

训练 GoogleNet 时遇到了典型的训练失败问题：
- **Loss 完全无法下降**：训练损失停滞不前，几乎没有变化
- **准确率极低**：始终徘徊在随机猜测水平（10分类约10%）
- **训练毫无进展**：无论调整学习率、batch size 都无效

> **状态：✅ 问题已解决** - 通过添加 Xavier 初始化彻底解决

## 相关代码

本次问题的完整代码实现：
- [GoogleNet 实现代码](https://github.com/Hao-yiwen/deeplearning/blob/master/pytorch_2025/month_11/chapter_5_cnn/practise_8_googlenet_no_d2l.ipynb)

## 问题根源（已定位）

**没有对网络进行 Xavier/Glorot 权重初始化**！

PyTorch 默认的初始化方式对于像 GoogleNet 这样的深层网络来说不够理想，导致训练从一开始就陷入困境。

## 为什么权重初始化如此重要？

### 1. 梯度消失/爆炸问题
- 没有合适的初始化，深层网络容易出现梯度消失或梯度爆炸
- 梯度消失：反向传播时梯度变得极小，导致前面层几乎无法更新
- 梯度爆炸：梯度值过大，导致权重更新不稳定

### 2. 激活函数饱和
- 不当的初始化可能导致激活函数输入值过大或过小
- ReLU 可能进入"死区"（输入为负，输出恒为0）
- 导致神经元失效，无法学习

### 3. 收敛速度
- 好的初始化能让网络从一个较好的起点开始训练
- 显著加快收敛速度，提高训练效率

## 解决方案（已实施）

通过添加 Xavier/Glorot 初始化完全解决了问题！

### Xavier 初始化原理
Xavier 初始化（也称 Glorot 初始化）根据输入和输出神经元数量来设定权重的初始值范围：

```
权重 ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
或
权重 ~ Normal(0, √(2/(fan_in + fan_out)))
```

其中：
- `fan_in`：输入神经元数量
- `fan_out`：输出神经元数量

### 实现方式

有几种方式可以添加初始化，这次使用的是网络构建后统一初始化：

```python
def init_weights(net):
    """对整个网络进行权重初始化"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# 构建网络后立即初始化
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
init_weights(net)
```

## 常用的初始化方法对比

| 初始化方法 | 适用场景 | PyTorch 函数 |
|-----------|---------|-------------|
| **Xavier/Glorot** | Sigmoid、Tanh 激活函数 | `nn.init.xavier_uniform_()` / `xavier_normal_()` |
| **He/Kaiming** | ReLU 及其变体 | `nn.init.kaiming_uniform_()` / `kaiming_normal_()` |
| **正态分布** | 通用，需手动调整标准差 | `nn.init.normal_(mean, std)` |
| **均匀分布** | 通用 | `nn.init.uniform_(a, b)` |

## 经验教训

### 问题症状（解决前）
- ✗ Loss 在前几个 epoch 就停滞不动
- ✗ 准确率接近随机猜测（10分类约 10%）
- ✗ 调整学习率、batch size 等超参数都无效

### 解决后的效果
- ✓ Loss 从第一个 epoch 就开始稳定下降
- ✓ 准确率快速提升
- ✓ 训练稳定，收敛速度明显加快

### 核心教训

**深度神经网络训练前，权重初始化是必需步骤！** 这个问题看起来简单，但极易被忽视：

1. **对于使用 ReLU 的网络**（如本次的 GoogleNet）
   - 应该使用 He/Kaiming 初始化，但 Xavier 初始化通常也能工作

2. **对于使用 Sigmoid/Tanh 的网络**
   - 必须使用 Xavier 初始化

3. **不要依赖 PyTorch 默认初始化**
   - 默认初始化对浅层网络可能够用
   - 对深层网络（尤其是 Inception 这种复杂结构）往往会失败

## 参考资料

- [动手学深度学习 - GoogLeNet](https://zh-v2.d2l.ai/chapter_convolutional-modern/googlenet.html)
- [完整代码实现](https://github.com/Hao-yiwen/deeplearning/blob/master/pytorch_2025/month_11/chapter_5_cnn/practise_8_googlenet_no_d2l.ipynb)
