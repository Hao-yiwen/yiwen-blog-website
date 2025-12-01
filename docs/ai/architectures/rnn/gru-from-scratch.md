---
title: GRU 从零实现
sidebar_label: GRU 从零实现
sidebar_position: 2
tags: [GRU, PyTorch, 深度学习, 序列模型]
---

# GRU (门控循环单元) 从零实现

GRU（Gated Recurrent Unit，门控循环单元）是 LSTM 的一个"简化进阶版"。

如果说普通 RNN 是"金鱼记忆"（容易忘事），LSTM 是"精密的文件柜"（功能强大但结构复杂），那么 **GRU 就是一个更轻量、更高效的现代版文件柜**。

## 为什么需要 GRU？

我们知道普通 RNN 有**梯度消失**的问题，无法捕捉长距离依赖。LSTM 通过引入三个"门"（输入门、遗忘门、输出门）解决了这个问题，但参数很多，计算慢。

**GRU (2014年提出)** 在保持 LSTM 记忆能力的同时，将结构简化为**两个门**：

| 门 | 名称 | 作用 |
|----|------|------|
| $r_t$ | 重置门 (Reset Gate) | 决定在计算新候选记忆时，要**忽略**多少之前的隐状态 |
| $z_t$ | 更新门 (Update Gate) | 决定要保留多少旧状态，以及要写入多少新状态 |

> **核心区别：** GRU 没有单独的"细胞状态 (Cell State)"，它的隐状态 $h_t$ 既负责记忆，也负责输出。

## 数学原理

在时间步 $t$，给定输入 $x_t$ 和上一时刻隐状态 $h_{t-1}$：

### 步骤 1：计算门控

使用 Sigmoid 函数 ($\sigma$) 将值压缩到 $[0, 1]$：

$$
r_t = \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r)
$$

$$
z_t = \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z)
$$

### 步骤 2：计算候选隐状态

这里使用重置门 $r_t$。如果 $r_t \approx 0$，意味着之前的记忆 $h_{t-1}$ 被"切断"，模型就像在处理序列的第一个词一样：

$$
\tilde{h}_t = \tanh(W_{in} x_t + r_t \odot (W_{hn} h_{t-1}) + b_n)
$$

*（$\odot$ 代表逐元素相乘）*

### 步骤 3：计算最终隐状态

利用更新门 $z_t$ 进行**线性插值**：
- 如果 $z_t \approx 1$：主要保留旧记忆
- 如果 $z_t \approx 0$：主要使用新的候选状态

$$
h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}
$$

## 信息流图示

```
                    ┌────────────────────────────────────────┐
                    │            GRU Cell                    │
                    │                                        │
  h_{t-1} ─────────▶│  ┌──────┐   ┌──────┐                  │
      │             │  │ r_t  │   │ z_t  │                  │
      │             │  │重置门│   │更新门│                  │
      │             │  └──┬───┘   └──┬───┘                  │
      │             │     │          │                      │
      │             │     ▼          │                      │
      │             │  ┌──────┐      │                      │
      │             │  │~h_t  │      │                      │──▶ h_t
      └─────────────│─▶│候选态│──────┴──────────────────────│
                    │  └──────┘      ▲                      │
                    │                │                      │
  x_t ──────────────│────────────────┘                      │
                    │                                        │
                    └────────────────────────────────────────┘
```

## PyTorch 从零实现

我们将手动实现上述逻辑，不使用 `nn.GRU`：

```python
import torch
import torch.nn as nn

class GRUFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUFromScratch, self).__init__()
        self.hidden_size = hidden_size

        # --- 1. 定义更新门 (Update Gate) 的权重 ---
        self.x2z = nn.Linear(input_size, hidden_size)   # x -> z
        self.h2z = nn.Linear(hidden_size, hidden_size)  # h -> z

        # --- 2. 定义重置门 (Reset Gate) 的权重 ---
        self.x2r = nn.Linear(input_size, hidden_size)   # x -> r
        self.h2r = nn.Linear(hidden_size, hidden_size)  # h -> r

        # --- 3. 定义候选状态 (Candidate State) 的权重 ---
        self.x2n = nn.Linear(input_size, hidden_size)   # x -> n
        self.h2n = nn.Linear(hidden_size, hidden_size)  # h -> n

        # --- 4. 输出层 ---
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        :param x: (batch_size, seq_len, input_size)
        :param hidden: (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []

        # === 时间步循环 ===
        for t in range(seq_len):
            x_t = x[:, t, :]

            # 1. 计算更新门 z_t
            z_t = torch.sigmoid(self.x2z(x_t) + self.h2z(hidden))

            # 2. 计算重置门 r_t
            r_t = torch.sigmoid(self.x2r(x_t) + self.h2r(hidden))

            # 3. 计算候选隐状态 n_t
            # r_t 作用于 h 经过线性变换后的结果
            h_reset = r_t * self.h2n(hidden)
            n_t = torch.tanh(self.x2n(x_t) + h_reset)

            # 4. 计算最终隐状态 h_t
            # 软切换：z_t 控制保留旧信息，(1-z_t) 控制接受新信息
            hidden = (1 - z_t) * n_t + z_t * hidden

            # 5. 计算输出
            out_t = self.output_layer(hidden)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden
```

### 测试代码

```python
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 5
BATCH_SIZE = 3
SEQ_LEN = 6

gru_model = GRUFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

output, final_hidden = gru_model(dummy_input)

print(f"GRU 输出形状: {output.shape}")         # [3, 6, 5]
print(f"GRU 最终隐状态: {final_hidden.shape}") # [3, 20]
```

输出：
```
GRU 输出形状: torch.Size([3, 6, 5])
GRU 最终隐状态: torch.Size([3, 20])
```

## 代码关键细节

### Sigmoid vs Tanh 的选择

| 组件 | 激活函数 | 原因 |
|------|----------|------|
| 门 ($z_t$, $r_t$) | Sigmoid | 门的物理意义是"比例/开关"，值必须在 $[0, 1]$ |
| 候选状态 ($\tilde{h}_t$) | Tanh | 真正的数据信息，值域 $[-1, 1]$，保持梯度稳定 |

### 软门控 (Soft Gating)

```python
hidden = (1 - z_t) * n_t + z_t * hidden
```

这行代码完全可微。模型可以通过反向传播自己学习：
- 遇到句号时，$z_t \approx 0$（全更新，遗忘上一句）
- 遇到连词时，$z_t \approx 1$（保持记忆）

## GRU vs LSTM 对比

| 特性 | GRU | LSTM |
|------|-----|------|
| **门数量** | 2 个 | 3 个 + 细胞状态 |
| **参数量** | 少（约为 LSTM 的 75%） | 多 |
| **训练速度** | 快 | 稍慢 |
| **表现** | 小数据集/短序列往往更优 | 超长序列/大数据集潜力更大 |

**经验法则：** 实践中通常**先尝试 GRU**（训练快，效果通常和 LSTM 差不多）。如果 GRU 效果遇到瓶颈，或者需要处理非常复杂的长依赖关系，再切换到 LSTM。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) - GRU 原始论文
- [PyTorch GRU 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
