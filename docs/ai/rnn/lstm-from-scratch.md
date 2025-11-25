---
title: LSTM 从零实现
sidebar_label: LSTM 从零实现
sidebar_position: 3
tags: [LSTM, PyTorch, 深度学习, 序列模型]
---

# LSTM (长短期记忆网络) 从零实现

LSTM (Long Short-Term Memory, 长短期记忆网络) 是序列模型之旅的"最终Boss"。

如果说 RNN 是记性不好的"金鱼"，GRU 是高效的"现代文件柜"，那么 LSTM 就是一条**精密控制的信息高速公路**。

它是目前最经典、应用最广泛的循环网络变体，专门设计用于解决长序列训练中的梯度消失问题，能够捕捉非常长期的依赖关系。

## 核心概念：细胞状态 (Cell State)

LSTM 与 RNN/GRU 最大的不同在于，它在每个时间步维护**两个**状态：

| 状态 | 符号 | 作用 |
|------|------|------|
| 隐状态 | $h_t$ | 当前时刻的短期工作记忆，同时作为输出 |
| 细胞状态 | $C_t$ | 长期记忆，像一条贯穿整个时间序列的"信息高速公路" |

> **核心直觉：** 信息在高速公路 $C_t$ 上流动时，只有少量的线性交互。这使得信息很容易保持不变地流过很长的距离。LSTM 通过精心设计的"门"结构，来控制何时向这条高速公路上添加信息，或者何时从上面移除信息。

## 三个"门"的精密运作

LSTM 在每个时间步 $t$，接收输入 $x_t$ 以及上一时刻的两个状态 $(h_{t-1}, C_{t-1})$。

*（所有门都使用 Sigmoid 激活函数 $\sigma$，输出 0 到 1 之间的值，表示"通过率"）*

### 步骤 1：遗忘门 (Forget Gate)

**"决定要忘记什么"**

它查看当前的输入 $x_t$ 和上一个短时记忆 $h_{t-1}$，为上一个长时记忆 $C_{t-1}$ 中的每个数字输出一个 $0$ 到 $1$ 之间的值：
- $1$ 代表"完全保留"
- $0$ 代表"完全遗忘"

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

### 步骤 2：输入门 (Input Gate)

**"决定要存储什么新信息"**

这一步分为两个子步骤：

1. **输入门 ($i_t$)：** 决定我们将更新哪些值
2. **候选记忆 ($\tilde{C}_t$)：** 创建一个新的候选值向量

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

### 步骤 3：更新细胞状态

**"执行遗忘和记忆"** —— 这是最关键的一步！

我们将旧的"高速公路"状态 $C_{t-1}$ 更新为新的 $C_t$：
- 把旧状态乘以遗忘门 $f_t$（忘掉要丢弃的信息）
- 加上输入门 $i_t$ 和候选记忆 $\tilde{C}_t$ 的乘积（加入新信息）

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

*（$\odot$ 为逐元素乘法。这个加法操作是梯度能够长距离传播的关键！）*

### 步骤 4：输出门 (Output Gate)

**"决定基于当前状态输出什么"**

1. 运行输出门 $o_t$，决定细胞状态的哪些部分将用于输出
2. 将细胞状态 $C_t$ 通过 $\tanh$，然后与输出门相乘

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

## LSTM 结构图示

```
                         C_{t-1} ─────────────────────────────────────▶ C_t
                            │              ×           +              │
                            │              │           │              │
                            ▼              │           │              │
┌───────────────────────────────────────────────────────────────────────────────┐
│                                    LSTM Cell                                   │
│                                                                               │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│    │  f_t    │    │  i_t    │    │ ~C_t    │    │  o_t    │                  │
│    │ 遗忘门  │    │ 输入门  │    │ 候选态  │    │ 输出门  │                  │
│    │   σ     │    │   σ     │    │  tanh   │    │   σ     │                  │
│    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘                  │
│         │              │              │              │                        │
│         └──────────────┴──────────────┴──────────────┘                        │
│                                 ▲                                             │
│                                 │                                             │
│                    ┌────────────┴────────────┐                                │
│                    │    [h_{t-1}, x_t]       │                                │
│                    └────────────────────────┘                                │
└───────────────────────────────────────────────────────────────────────────────┘
                                 │
   h_{t-1} ◀─────────────────────┴─────────────────────────────────────▶ h_t
```

## PyTorch 从零实现

为了效率，实际框架中通常**将 4 个门的操作合并为一个大的矩阵乘法**，然后再将结果切分。这在数学上等价，但计算效率更高。

```python
import torch
import torch.nn as nn

class LSTMFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMFromScratch, self).__init__()
        self.hidden_size = hidden_size

        # --- 定义权重 ---
        # 将4个门的权重合并：输出维度是 4 * hidden_size
        # 顺序: input gate (i), forget gate (f), candidate (g), output gate (o)

        # 处理输入 x 的大矩阵: x -> (i, f, g, o)
        self.x2gates = nn.Linear(input_size, 4 * hidden_size)
        # 处理隐状态 h 的大矩阵: h -> (i, f, g, o)
        self.h2gates = nn.Linear(hidden_size, 4 * hidden_size)

        # 最终任务的输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        """
        :param x: (batch_size, seq_len, input_size)
        :param hidden_state: 元组 (h_0, c_0)，每个形状 (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化 h_0 和 c_0
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden_state

        outputs = []

        # === 时间步循环 ===
        for t in range(seq_len):
            x_t = x[:, t, :]

            # --- 计算所有门控的预激活值 ---
            # gates_pre 形状: (batch_size, 4 * hidden_size)
            gates_pre = self.x2gates(x_t) + self.h2gates(h_t)

            # --- 切分为 4 份 ---
            # 每块形状: (batch_size, hidden_size)
            i_pre, f_pre, g_pre, o_pre = gates_pre.chunk(4, dim=1)

            # --- 应用激活函数 ---
            i_t = torch.sigmoid(i_pre)  # 输入门
            f_t = torch.sigmoid(f_pre)  # 遗忘门
            g_t = torch.tanh(g_pre)     # 候选细胞状态
            o_t = torch.sigmoid(o_pre)  # 输出门

            # --- 更新细胞状态 C_t (核心步骤) ---
            c_t = f_t * c_t + i_t * g_t

            # --- 更新隐状态 h_t ---
            h_t = o_t * torch.tanh(c_t)

            # --- 计算当前步输出 ---
            out_t = self.output_layer(h_t)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)

        # 返回所有步的输出，以及最后的 (h_n, c_n) 状态元组
        return outputs, (h_t, c_t)
```

### 测试代码

```python
INPUT_SIZE = 10
HIDDEN_SIZE = 20
OUTPUT_SIZE = 5
BATCH_SIZE = 3
SEQ_LEN = 6

lstm_model = LSTMFromScratch(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)

# 注意：LSTM 返回的最终状态是一个元组 (h_n, c_n)
output, (final_h, final_c) = lstm_model(dummy_input)

print(f"LSTM 输出形状: {output.shape}")          # [3, 6, 5]
print(f"LSTM 最终隐状态 h 形状: {final_h.shape}") # [3, 20]
print(f"LSTM 最终细胞状态 c 形状: {final_c.shape}") # [3, 20]
```

输出：
```
LSTM 输出形状: torch.Size([3, 6, 5])
LSTM 最终隐状态 h 形状: torch.Size([3, 20])
LSTM 最终细胞状态 c 形状: torch.Size([3, 20])
```

## 为什么 LSTM 能解决梯度消失？

关键在于细胞状态的更新公式：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

这是一个**加法**操作！

在反向传播时，梯度通过加法几乎无损地传递。只要遗忘门 $f_t$ 不完全关闭（$f_t > 0$），梯度就能沿着细胞状态"高速公路"一路回传到很早的时间步。

## 三种模型总结对比

| 模型 | 状态数量 | 门数量 | 核心机制 | 适用场景 |
|------|:--------:|:------:|----------|----------|
| **RNN** | 1 ($h_t$) | 0 | 简单线性变换 + 激活 | 极短序列，教学示例 |
| **GRU** | 1 ($h_t$) | 2 | 门控机制控制信息流 | 大多数任务首选，速度快 |
| **LSTM** | 2 ($h_t, C_t$) | 3 | **细胞状态高速公路** | 超长序列，复杂依赖关系 |

理解了这些手动实现的代码，再回头去看 PyTorch 官方文档中的 `nn.RNN`, `nn.GRU`, `nn.LSTM`，你会发现它们不再是黑盒子，而是你完全理解的数学计算过程。

## 参考资料

- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - LSTM 原始论文 (1997)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah 的经典博客
- [PyTorch LSTM 官方文档](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
