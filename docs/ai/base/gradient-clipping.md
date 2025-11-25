---
title: 梯度裁剪（Gradient Clipping）
sidebar_position: 15
tags: [深度学习, 梯度裁剪, 梯度爆炸, 训练技巧]
---

# 梯度裁剪（Gradient Clipping）

## 什么是梯度裁剪？

梯度裁剪是一种防止**梯度爆炸**的技术。在深度神经网络训练过程中，梯度可能会变得非常大，导致参数更新过大，使训练不稳定甚至发散。梯度裁剪通过限制梯度的大小来解决这个问题。

## 为什么需要梯度裁剪？

### 梯度爆炸问题

在深度网络中，尤其是 RNN/LSTM 等序列模型中，反向传播时梯度会逐层累乘：

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdot ... \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}
$$

如果每一层的梯度 $> 1$，经过多层累乘后，梯度会指数级增长，导致：
- 参数更新过大
- 损失函数震荡或发散
- 出现 NaN 值

### 典型场景

- **RNN/LSTM/GRU**：处理长序列时特别容易出现梯度爆炸
- **深层网络**：层数越多，梯度累乘越严重
- **Transformer**：在某些配置下也可能出现

## 梯度裁剪的两种方法

### 1. 按值裁剪（Clip by Value）

直接将梯度限制在 $[-threshold, threshold]$ 范围内：

$$
g_i = \begin{cases}
threshold & \text{if } g_i > threshold \\
-threshold & \text{if } g_i < -threshold \\
g_i & \text{otherwise}
\end{cases}
$$

```python
import torch

# PyTorch 实现
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

**特点**：
- 简单直接
- 可能改变梯度方向
- 不常用

### 2. 按范数裁剪（Clip by Norm）⭐ 推荐

计算所有梯度的全局范数，如果超过阈值则等比例缩放：

$$
\|g\| = \sqrt{\sum_i g_i^2}
$$

$$
g = \begin{cases}
g \cdot \frac{max\_norm}{\|g\|} & \text{if } \|g\| > max\_norm \\
g & \text{otherwise}
\end{cases}
$$

```python
import torch

# PyTorch 实现（最常用）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**特点**：
- 保持梯度方向不变
- 只缩放大小
- **业界标准做法**

## PyTorch 完整示例

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # 梯度裁剪（在 optimizer.step() 之前）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
```

## 如何选择裁剪阈值？

### 常用经验值

| 模型类型 | 推荐阈值 |
|---------|---------|
| RNN/LSTM/GRU | 1.0 - 5.0 |
| Transformer | 1.0 |
| CNN | 通常不需要 |
| 一般深度网络 | 1.0 - 10.0 |

### 动态调整方法

```python
# 监控梯度范数
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"Gradient norm: {total_norm}")

# 根据观察到的范数选择合适的阈值
```

## 梯度裁剪 vs 其他技术

| 技术 | 解决问题 | 原理 |
|-----|---------|------|
| **梯度裁剪** | 梯度爆炸 | 限制梯度大小 |
| **BatchNorm** | 内部协变量偏移 | 归一化激活值 |
| **权重初始化** | 梯度消失/爆炸 | 合理初始化参数 |
| **残差连接** | 梯度消失 | 跳跃连接 |
| **学习率调度** | 训练不稳定 | 动态调整学习率 |

## 注意事项

1. **裁剪时机**：必须在 `loss.backward()` 之后、`optimizer.step()` 之前
2. **不要过度裁剪**：阈值太小会导致训练变慢
3. **监控梯度**：训练时记录梯度范数，帮助诊断问题
4. **与其他技术配合**：梯度裁剪通常与学习率调度、权重衰减等一起使用

## 总结

- 梯度裁剪是防止梯度爆炸的有效手段
- **按范数裁剪**是推荐做法，保持梯度方向不变
- RNN/LSTM 等序列模型几乎必须使用梯度裁剪
- 常用阈值范围：1.0 - 5.0

## 参考资源

- [PyTorch clip_grad_norm_ 文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [反向传播算法](./backpropagation.md)
- [参数初始化指南](./parameter-initialization-guide.md)
