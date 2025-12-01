---
title: RNN 从零实现
sidebar_label: RNN 从零实现
sidebar_position: 1
tags: [RNN, PyTorch, 深度学习, 序列模型]
---

# 循环神经网络 (RNN) 从零实现

理解循环神经网络（RNN）的内部机制是掌握深度学习序列模型（如 LSTM、Transformer）的基石。本文将从直观和数学角度介绍 RNN，然后使用 PyTorch **从零实现**一个 RNN 模型。

## 为什么需要 RNN？

传统的神经网络（如全连接层或 CNN）假设输入是相互独立的。例如，当你向网络输入一张猫的照片时，网络并不关心上一张照片是什么。

但在处理**序列数据**（如文本、音频、股票价格、天气预报）时，前后文是有关系的：

- **天气预报**：明天的天气往往受前几天天气影响
- **自然语言**：理解句子中的"苹果"是指水果还是手机，往往取决于前面的词
- **时间序列**：股票价格与历史趋势密切相关

RNN 的核心思想就是引入**记忆（Memory）**，使网络能够保持和利用历史信息。

## 核心机制：隐状态 (Hidden State)

RNN 在处理序列中的每个元素时，不仅输入当前时刻的数据 $x_t$，还会输入上一个时刻的**隐状态 $h_{t-1}$**。这个隐状态包含了之前所有时刻的信息摘要。

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌───┐      ┌───┐      ┌───┐      ┌───┐      ┌───┐       │
│    │h_0│─────▶│h_1│─────▶│h_2│─────▶│h_3│─────▶│h_4│       │
│    └───┘      └───┘      └───┘      └───┘      └───┘       │
│      ▲          ▲          ▲          ▲          ▲         │
│      │          │          │          │          │         │
│    ┌───┐      ┌───┐      ┌───┐      ┌───┐      ┌───┐       │
│    │x_0│      │x_1│      │x_2│      │x_3│      │x_4│       │
│    └───┘      └───┘      └───┘      └───┘      └───┘       │
│                                                             │
│              隐状态在时间步之间传递信息                        │
└─────────────────────────────────────────────────────────────┘
```

## 数学公式

### 隐藏状态更新

对于时间步 $t$，RNN 的核心计算公式如下：

$$
h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})
$$

其中：

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前时刻的输入向量 |
| $h_{t-1}$ | 上一时刻的隐状态 |
| $h_t$ | 当前时刻计算出的新隐状态 |
| $W_{ih}$ | 输入到隐层的权重矩阵 |
| $W_{hh}$ | 隐层到隐层的权重矩阵（记忆的权重） |
| $b_{ih}, b_{hh}$ | 偏置项 |
| $\tanh$ | 激活函数，将值压缩到 $[-1, 1]$ |

### 输出计算

输出 $y_t$ 通常由当前的 $h_t$ 经过另一个线性变换得到：

$$
y_t = W_{ho} h_t + b_{ho}
$$

其中：
- $y_t$：当前时刻输出
- $W_{ho}$：输出权重矩阵
- $b_{ho}$：输出偏置项

## 网络特点

| 特点 | 说明 |
|------|------|
| **循环连接** | 引入循环连接，使网络具有"记忆"能力 |
| **参数共享** | 各时刻使用相同的权重矩阵，模型参数量不随序列长度增加 |
| **变长序列** | 能够处理变长序列输入 |
| **时序建模** | 适合建模序列数据的时序依赖关系 |

## PyTorch 从零实现

为了彻底理解 RNN，我们手动实现其核心逻辑：**在时间维度上进行 `for` 循环**，而不是直接调用 `nn.RNN`。

```python
import torch
import torch.nn as nn

class RNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 RNN 的权重
        :param input_size: 输入特征的维度 (例如词向量维度)
        :param hidden_size: 隐状态/记忆的维度
        :param output_size: 输出的维度 (例如分类数)
        """
        super(RNNFromScratch, self).__init__()

        self.hidden_size = hidden_size

        # 1. 定义输入到隐层的权重 (对应 W_ih)
        self.i2h = nn.Linear(input_size, hidden_size)

        # 2. 定义隐层到隐层的权重 (对应 W_hh)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        # 3. 定义隐层到输出的权重 (对应 W_ho)
        self.h2o = nn.Linear(hidden_size, output_size)

        # 4. 激活函数
        self.activation = nn.Tanh()

    def forward(self, x, hidden=None):
        """
        前向传播逻辑
        :param x: 输入数据，形状为 (batch_size, seq_len, input_size)
        :param hidden: 初始隐状态，形状为 (batch_size, hidden_size)
        :return: 所有时间步的输出, 最后的隐状态
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 如果没有提供初始隐状态，则初始化为全 0
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []

        # === 核心循环：遍历时间序列 ===
        for t in range(seq_len):
            # 获取当前时刻的输入 x_t: (batch_size, input_size)
            x_t = x[:, t, :]

            # === RNN 公式实现 ===
            # h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
            # 注意：nn.Linear 内部已经包含了偏置 b
            i2h_val = self.i2h(x_t)
            h2h_val = self.h2h(hidden)

            # 更新隐状态
            hidden = self.activation(i2h_val + h2h_val)

            # 计算当前时刻的输出
            out_t = self.h2o(hidden)
            outputs.append(out_t)

        # 将列表转换为张量: (batch_size, seq_len, output_size)
        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden
```

### 测试代码

```python
# 设定参数
INPUT_SIZE = 10   # 每个时刻输入一个长度为 10 的向量
HIDDEN_SIZE = 20  # 记忆容量
OUTPUT_SIZE = 5   # 5 分类任务
BATCH_SIZE = 3    # 一次处理 3 个样本
SEQ_LEN = 6       # 序列长度为 6 (例如一句话有 6 个词)

# 实例化模型
rnn = RNNFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# 创建随机输入数据
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

# 前向传播
output, final_hidden = rnn(dummy_input)

print(f"输入形状: {dummy_input.shape}")        # [3, 6, 10]
print(f"输出形状: {output.shape}")             # [3, 6, 5] -> (Batch, Seq, Output)
print(f"最终隐状态形状: {final_hidden.shape}") # [3, 20] -> (Batch, Hidden)
```

输出：
```
输入形状: torch.Size([3, 6, 10])
输出形状: torch.Size([3, 6, 5])
最终隐状态形状: torch.Size([3, 20])
```

## 代码关键点解析

### 1. 时间步循环

```python
for t in range(seq_len):
    x_t = x[:, t, :]
    # ... 处理当前时刻
```

这里的 `for` 循环是 RNN 的灵魂，它显式地展示了网络是如何一步一步"阅读"序列的。

### 2. 权重共享

`self.i2h` 和 `self.h2h` 是在 `__init__` 中定义的。在整个时间步循环中（$t=0$ 到 $t=5$），我们使用的是**同一组**权重矩阵。**无论序列多长，模型参数量不变**。

### 3. 状态传递

`hidden` 变量在循环外部初始化，在循环内部更新，并传递给下一次循环。这就是信息流动的载体：

```python
hidden = torch.zeros(...)  # 初始化
for t in range(seq_len):
    hidden = self.activation(...)  # 更新并传递
```

## 简单 RNN 的局限性

虽然上面的代码完美展示了 RNN 的原理，但在实际应用中，普通的 RNN（Vanilla RNN）有两个严重问题：

### 梯度消失 (Vanishing Gradient)

当序列很长时，反向传播的梯度在经过多次 $\tanh$ 导数乘法后会趋近于 0。

**结果**：模型"忘记"很久以前的信息（比如读到段落结尾忘了开头的主语）。

### 梯度爆炸 (Exploding Gradient)

梯度也可能变得极大，导致权重更新时数值溢出。

**结果**：训练不稳定，loss 变成 NaN。

### 解决方案

实际工程中，我们通常使用更高级的变体：

| 模型 | 特点 |
|------|------|
| **LSTM** (长短期记忆网络) | 引入门控机制（遗忘门、输入门、输出门），有选择性地保留或遗忘信息 |
| **GRU** (门控循环单元) | LSTM 的简化版本，参数更少，性能相近 |

## 参考资料

- [PyTorch RNN 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [字符级 RNN 实践](https://github.com/Hao-yiwen/deeplearning/blob/master/pytorch/week3/practise_7_char_rnn.ipynb)
