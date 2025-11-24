---
title: 线性回归反向传播推导
sidebar_label: 线性回归反向传播推导
date: 2025-11-24
tags: [深度学习, 反向传播, 线性回归, 梯度下降, PyTorch]
---

# 线性回归反向传播推导

线性回归是机器学习中最基础的模型之一。虽然 PyTorch 的 `backward()` 方法能自动完成复杂的计算图反向传播，但手动推导梯度计算的数学过程，对于理解深度学习的核心原理至关重要。

本文将详细推导线性回归模型中 `l.sum().backward()` 背后的数学原理。

## 1. 数学公式定义

### 1.1 前向传播（Forward Propagation）

**模型定义：**

$$
\hat{y} = Xw + b
$$

其中：
- $X$：形状为 $(N, d)$ 的输入矩阵（$N$ 为 batch size，$d$ 为特征维度）
- $w$：形状为 $(d, 1)$ 的权重向量
- $b$：标量偏置
- $\hat{y}$：形状为 $(N, 1)$ 的预测值

### 1.2 损失函数（Loss Function）

**单样本损失（均方误差）：**

$$
l^{(i)} = \frac{1}{2} (\hat{y}^{(i)} - y^{(i)})^2
$$

注意：系数 $\frac{1}{2}$ 是为了求导时消除常数系数，简化梯度计算。

**总损失（Batch Loss）：**

当执行 `l.sum().backward()` 时，反向传播的起点是所有样本损失的**求和**（而非平均）：

$$
L = \sum_{i=1}^{N} l^{(i)} = \sum_{i=1}^{N} \frac{1}{2} (\hat{y}^{(i)} - y^{(i)})^2
$$

:::tip 为什么用 sum() 而不是 mean()？
在 PyTorch 中，`backward()` 方法需要从一个标量开始反向传播。使用 `sum()` 计算总梯度后，在参数更新时除以 `batch_size`，在数学上等价于直接使用平均损失，但代码实现更加灵活。
:::

## 2. 梯度推导（链式法则）

### 2.1 预测值的梯度

首先计算损失函数 $L$ 对预测值 $\hat{y}$ 的偏导数：

$$
\frac{\partial L}{\partial \hat{y}^{(i)}} = \frac{\partial}{\partial \hat{y}^{(i)}} \left[ \frac{1}{2}(\hat{y}^{(i)} - y^{(i)})^2 \right] = (\hat{y}^{(i)} - y^{(i)})
$$

定义**误差项**：

$$
\delta^{(i)} = \hat{y}^{(i)} - y^{(i)}
$$

### 2.2 权重 $w$ 的梯度

根据**链式法则**：

$$
\frac{\partial L}{\partial w} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial w}
$$

由于 $\hat{y}^{(i)} = x^{(i)} w + b$，我们有：

$$
\frac{\partial \hat{y}^{(i)}}{\partial w} = x^{(i)}
$$

代入得到：

$$
\frac{\partial L}{\partial w} = \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}
$$

**矩阵形式**（PyTorch 实际计算方式）：

$$
\frac{\partial L}{\partial w} = X^T (\hat{y} - y)
$$

其中：
- $(\hat{y} - y)$ 是形状为 $(N, 1)$ 的误差向量
- $X^T$ 是形状为 $(d, N)$ 的转置矩阵
- 结果是 $(d, 1)$ 的梯度向量，与 $w$ 形状一致

### 2.3 偏置 $b$ 的梯度

同样应用**链式法则**：

$$
\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial b}
$$

由于 $\frac{\partial \hat{y}^{(i)}}{\partial b} = 1$，因此：

$$
\frac{\partial L}{\partial b} = \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})
$$

:::info 关键洞察
偏置 $b$ 的梯度就是**所有样本误差的简单求和**。
:::

## 3. PyTorch 代码实现分析

### 3.1 基础代码示例

```python
import torch

# 定义线性回归模型
def net(X, w, b):
    return torch.matmul(X, w) + b

# 均方误差损失函数
def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2

# 随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练过程
for X, y in data_iter:
    l = squared_loss(net(X, w, b), y)  # 计算损失
    l.sum().backward()  # 反向传播
    sgd([w, b], lr, batch_size)  # 更新参数
```

### 3.2 关键步骤解析

#### Step 1: 计算梯度

```python
l.sum().backward()
```

- 计算所有样本的**总梯度**（而非平均梯度）
- 梯度存储在 `w.grad` 和 `b.grad` 中

#### Step 2: 参数更新

```python
param -= lr * param.grad / batch_size
```

- 除以 `batch_size` 实现"平均梯度下降"
- 数学等价于：$w \leftarrow w - \text{lr} \times \frac{\sum \text{Gradients}}{N}$

:::caution 为什么要除以 batch_size？
如果不除以 batch size，梯度会随着批次大小增大而变大，导致训练不稳定甚至梯度爆炸。通过除以 batch size，我们得到的是**平均梯度**，使得不同批次大小下的训练行为更加一致。
:::

## 4. 符号含义详解

在机器学习公式中，符号有严格的约定：

| 符号 | 名称 | 含义 | 代码对应 |
|------|------|------|----------|
| $y$ | 标签 (Label) / 真实值 (Ground Truth) | 数据集中的**正确答案** | `labels` 或 `y` |
| $\hat{y}$ | 预测值 (Prediction) | 模型**计算出的值** | `net(X, w, b)` |
| $X$ | 特征 (Features) | 输入数据 | `X` |
| $w$ | 权重 (Weights) | 模型参数 | `w` |
| $b$ | 偏置 (Bias) | 模型参数 | `b` |

### 4.1 实例说明：房价预测

假设训练一个模型，通过"房屋面积"预测"房价"：

- **$X$ (输入)**：房屋面积 = 100平米
- **$y$ (真实值)**：实际售价 = 200万（**标准答案**）
- **$\hat{y}$ (预测值)**：模型预测 = 180万（**模型猜测**）

**损失计算：**

$$
l = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(180 - 200)^2 = 200
$$

- $\hat{y}$ 和 $y$ 越接近，损失越小，模型越准确
- 损失越大，反向传播时对参数的"惩罚"越大，更新幅度越大

## 5. 反向传播总结

### 5.1 计算流程

1. **前向传播**：$\hat{y} = Xw + b$
2. **计算损失**：$L = \sum_{i=1}^{N} \frac{1}{2}(\hat{y}^{(i)} - y^{(i)})^2$
3. **计算误差**：$\delta = \hat{y} - y$
4. **计算梯度**：
   - $\frac{\partial L}{\partial w} = X^T \delta$
   - $\frac{\partial L}{\partial b} = \sum \delta$
5. **更新参数**：
   - $w \leftarrow w - \frac{\text{lr}}{N} \cdot \frac{\partial L}{\partial w}$
   - $b \leftarrow b - \frac{\text{lr}}{N} \cdot \frac{\partial L}{\partial b}$

### 5.2 核心概念

:::tip 反向传播的本质
1. 计算预测值和真实值的**误差** $(\hat{y} - y)$
2. 通过链式法则，将误差**反向传播**到各个参数
3. **$w$ 的梯度** = 误差 × 输入特征 $X$
4. **$b$ 的梯度** = 误差求和
5. 梯度存储在 `w.grad` 和 `b.grad` 中，供优化器使用
:::

## 6. 数学推导的意义

理解反向传播的数学推导有助于：

1. **调试模型**：当梯度异常时，能够定位问题所在
2. **设计损失函数**：知道如何设计可求导的损失函数
3. **优化性能**：理解计算瓶颈，优化训练效率
4. **扩展到深层网络**：线性回归的推导是理解深度神经网络反向传播的基础

虽然现代深度学习框架（如 PyTorch、TensorFlow）提供了自动微分功能，但理解底层数学原理依然是成为优秀机器学习工程师的必经之路。

## 参考资源

- [PyTorch Autograd 机制](./autograd-mechanism.md)
- [反向传播算法](./backpropagation.md)
- [前向传播](./forwardpropagation.md)
