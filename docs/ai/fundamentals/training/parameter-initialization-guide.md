---
title: 参数初始化实用指南
sidebar_label: 参数初始化指南
sidebar_position: 15
date: 2025-11-16
tags: [深度学习, 参数初始化, 最佳实践, Transformer, RNN, CNN]
---

# 参数初始化实用指南

## 1. 为什么需要参数初始化？

神经网络训练一开始，所有权重都是随机的。
初始化做不好会导致：

* **梯度消失**：信号 / 梯度越传越小，深层几乎不更新
* **梯度爆炸**：数值越传越大，loss/梯度直接 NaN
* **训练很慢或很难收敛**

初始化的目标大致是：

> 让各层的输出和梯度在前向 / 反向传播中**方差保持相对稳定**。

---

## 2. 常见初始化方法

### 2.1 零 / 常数初始化（不推荐给权重）

* `W = 0` 或 `W = c`
* 只适合 **bias**（偏置），**不适合权重**：

  * 如果权重全是 0，每个神经元梯度完全一样，模型学不会有用表示。

典型用法：

* `bias = 0`（线性层 / 卷积层常见默认做法）

---

### 2.2 普通随机初始化（Uniform / Normal）

* 正态：`W ~ N(0, σ²)`
* 均匀：`W ~ U(-a, a)`

现在**很少直接"裸用"**，而是配合 fan_in / fan_out 设计成 Xavier / He 等。

---

### 2.3 Xavier 初始化（Glorot Initialization）

适用于：**激活函数为 tanh / sigmoid / 线性等对称、不过度截断的情况**

目标：让**前向和反向的方差大致一致**。

公式（简化）：

* Normal 版本：
  $$
  Var(W) = \frac{2}{n_{in} + n_{out}}
  $$
* Uniform 版本：
  $$
  W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\;\sqrt{\frac{6}{n_{in}+n_{out}}}\right)
  $$

在框架中的用法（PyTorch 示例）：

```python
import torch.nn as nn
import torch.nn.init as init

layer = nn.Linear(128, 256)
init.xavier_uniform_(layer.weight)
```

---

### 2.4 He 初始化（Kaiming Initialization）

适用于：**ReLU / LeakyReLU / GELU 等 ReLU 系列激活**

核心思想：ReLU 会把方差压缩大约一半，需要把权重方差设大一点补回来。

典型公式（Normal）：

$$
Var(W) = \frac{2}{n_{in}}
$$

PyTorch 示例：

```python
layer = nn.Linear(128, 256)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

实际工程中：

* CNN + ReLU
* MLP + ReLU / GELU

都优先考虑 He 初始化。

---

### 2.5 正交初始化（Orthogonal）

常用于：**RNN / LSTM / GRU 的循环权重**，提高长序列稳定性。

特点：

* 初始化得到的矩阵满足 $W^T W = I$
* 可以在一定程度上缓解 RNN 中梯度消失 / 爆炸

PyTorch 示例：

```python
W = torch.empty(128, 128)
nn.init.orthogonal_(W)
```

---

### 2.6 Transformer / LLM 中的初始化（简要）

现代 Transformer/大模型（如 GPT、LLaMA、Qwen 等）会使用**定制初始化 + 深度相关缩放**，不再是简单的 "Xavier/He 一把梭"。

常见做法（概念层面）：

* **Embedding / 线性层**：
  $$
  W \sim \mathcal{N}(0, \frac{1}{d_{model}})
  $$
* **注意力 Q/K/V 投影**：标准差通常设为 $1/\sqrt{d_{model}}$，控制 $QK^T$ 的尺度
* **残差路径缩放**：按层数 L 进行缩放，比如权重或输出乘以 $1/\sqrt{L}$，避免深层残差累计爆炸
* **LayerNorm**：γ=1，β=0，几乎是统一标准配置

你可以简单理解为：

> Transformer 用的是"基于 1/√d 和深度缩放的 Xavier 变体"，
> 专门为**深层残差 + 注意力结构**设计，比普通 He/Xavier 复杂一些。

---

### 2.7 LSTM / GRU 的偏置初始化

常见技巧：

* **遗忘门 bias 初始化为正值（比如 1）**，让模型一开始更倾向"记住信息"，稳定训练：

```python
for name, param in lstm.named_parameters():
    if "bias" in name:
        # PyTorch 中 LSTM bias 按门拼在一起，这里略写
        param.data.fill_(0.)
        # 其中 forget gate 对应那一段加 1
```

---

## 3. 实战中的简单建议（工程向）

如果你只是想"写代码时不要踩坑"，可以记住下面这几条：

1. **MLP / CNN + ReLU / GELU：**

   * 权重：He 初始化（`kaiming_normal_`）
   * bias：0

2. **RNN / LSTM / GRU：**

   * 输入权重：Xavier 或 He 都行
   * 循环权重：Orthogonal
   * forget gate bias：设为 1

3. **Transformer / LLM：**

   * 用框架 / 官方实现默认初始化
   * 如果自己写：线性层和 embedding 用 `std = 1/√d_model` 的 normal，再配合 LayerNorm + 残差缩放

4. **有 BatchNorm / LayerNorm / 残差 时：**

   * 初始化的重要性仍然存在，但比"纯深 MLP"更不敏感
   * 一般使用框架默认初始化就可以正常训练

---

## 参考资料

* [PyTorch 初始化文档](https://pytorch.org/docs/stable/nn.init.html)
* [Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)](http://proceedings.mlr.press/v9/glorot10a.html)
* [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852)
