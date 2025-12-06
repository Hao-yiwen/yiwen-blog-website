---
title: RoPE 旋转位置编码
sidebar_label: RoPE 旋转位置编码
date: 2025-12-06
last_update:
  date: 2025-12-06
tags: [transformer, positional-encoding, rope, llama]
---

# RoPE 旋转位置编码

在 Transformer 的架构演进中，位置编码（Positional Embedding）一直是核心话题。从最初的正弦位置编码（Sinusoidal），到可学习的绝对位置编码，再到如今 LLaMA、Mistral 等主流大模型标配的 **RoPE (Rotary Positional Embedding)**，我们一直在寻找一种更优雅的方式告诉模型"我是第几个字"。

RoPE 之所以成为主流，是因为它用一种极其巧妙的**几何**手段，解决了"绝对位置"与"相对位置"的统一问题。

本文将带你从最直观的几何原理出发，推导其数学形式，并最终落实到一行行的 PyTorch 代码实现。

## 1. 核心直觉：为什么是"旋转"？

在 RoPE 之前，主流做法通常是 **相加**：

$$x_{input} + P_{pos}$$

即把位置向量加到词向量上。这就像给每个词贴了个"号码牌"。

而 RoPE 的做法是 **旋转**：
它不再是加法，而是将向量在一个高维空间中进行旋转。

### 想象一组时钟

假设我们将词向量的每两个维度看作一个二维平面（复平面）。

- **绝对位置是旋转角度：** 第 $m$ 个 Token，我们就把它在这个平面上逆时针旋转 $m \times \theta$ 度。
- **相对位置是夹角：**
  - Token $m$ 旋转了 $m\theta$。
  - Token $n$ 旋转了 $n\theta$。
  - 它们之间的**相对角度差**就是 $(m-n)\theta$。

当我们在 Attention 机制中计算 $Q$ 和 $K$ 的点积（Dot Product）时，点积的大小取决于向量的长度和**夹角**。

**关键点来了：**
由于 RoPE 保证了旋转后的夹角只与 $(m-n)$ 有关，那么 $Q$ 和 $K$ 的点积结果，就天然地包含了它们的**相对距离信息**。我们不需要显式地告诉模型"这两个字距离为 5"，模型通过计算点积，发现夹角变了，自然就感知到了距离。

## 2. 数学推导：从复数到实数

为了看清本质，我们先用复数（Complex Numbers）来推导，因为二维旋转在复数域最简单。

假设二维向量 $q$ 表示为复数，位置为 $m$，基准频率为 $\theta$。

### Step 1: 施加旋转

$$f(q, m) = q \cdot e^{im\theta}$$

同理，对于位置 $n$ 的 key 向量 $k$：

$$f(k, n) = k \cdot e^{in\theta}$$

### Step 2: 计算 Attention Score (内积)

Attention 的核心是计算 query 和 key 的相似度。在复数域中，我们计算 Hermite 内积：

$$
\begin{aligned}
\text{Score} &= f(q, m) \cdot f(k, n)^* \\
&= (q \cdot e^{im\theta}) \cdot (k \cdot e^{in\theta})^* \\
&= q \cdot e^{im\theta} \cdot k^* \cdot e^{-in\theta} \\
&= (q \cdot k^*) \cdot e^{i(m-n)\theta}
\end{aligned}
$$

### Step 3: 回到实数域

取实部（对应实数向量的点积），我们发现结果包含：

$$\text{Result} \propto \cos((m-n)\theta)$$

**结论：** 最终的计算结果只依赖于 $q$ 和 $k$ 的原始内容，以及它们的相对距离 $(m-n)$。这就是 RoPE 具备**相对位置编码（Relative Positional Encoding）**特性的数学证明。

## 3. 矩阵形式与工程优化

在实际代码中，我们不能直接用复数。我们需要把二维向量 $\begin{bmatrix} x \\ y \end{bmatrix}$ 旋转 $\theta$ 角，对应线性代数中的旋转矩阵：

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
$$

### 极其巧妙的变形

为了在 GPU 上高效计算，我们避免写原本的矩阵乘法，而是利用以下数学恒等式：

$$
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} \cos\theta + \begin{bmatrix} -y \\ x \end{bmatrix} \sin\theta
$$

仔细观察第二项：$\begin{bmatrix} -y \\ x \end{bmatrix}$ 实际上就是将原向量 $\begin{bmatrix} x \\ y \end{bmatrix}$ 旋转了 90 度（并取反）。

这意味着：**RoPE 可以通过两次向量的 element-wise 乘法和一次加法来实现，无需构建巨大的旋转矩阵。**

## 4. PyTorch 代码全解

下面是 LLaMA 等模型中通用的 RoPE 实现。我们将逐行拆解。

### 4.1 定义 RoPE 模块

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        # dim: 这里必须是 head_size (即 model_dim / num_heads)
        # 因为旋转是在每个 Head 内部独立进行的

        # 1. 计算不同维度的频率 θ
        # 公式: theta_i = base^(-2i/d)
        # 频率跨度从 1 到 1/10000，高频捕捉短距离，低频捕捉长距离
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 2. 预计算所有位置的 cos 和 sin
        # 这是一个缓存表，避免每次 forward 都重算
        t = torch.arange(max_seq_len).float()

        # 外积操作: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, inv_freq)

        # 3. 拼接: 为了配合代码实现，将 [cos, sin] 扩展到完整维度
        # 这里的 emb 形状变为 [seq_len, dim]
        # 注意：这里 cat 了两次，是为了让 dim 0 和 dim dim/2 共享同一个频率
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, num_heads, head_dim]
        if seq_len > self.cos_cached.shape[0]:
            # 如果推理长度超过缓存，需要重新计算或报错
            pass
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]
```

### 4.2 应用旋转 (核心逻辑)

这里就是上文提到的"巧妙变形"的代码实现。

```python
def rotate_half(x):
    """
    将向量 x 切分成两半 (x1, x2)
    然后重组为 (-x2, x1)
    对应数学公式中的 [-y, x] 部分
    """
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: [batch, seq_len, num_heads, head_dim]
    cos, sin: [seq_len, head_dim] (广播到 batch 和 heads)
    """
    # 核心公式: Result = x * cos + rotate_90(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

## 5. 灵魂追问：位置信息到底是怎么生效的？

很多同学看懂了代码，但依然有一个疑问：

> **"我们只是分别旋转了 Q 和 K，并没有算它们之间的距离，为什么模型就知道 1 和 1000 的关系了？"**

答案就在 **Attention 的计算公式**里：

```python
# 当这行代码执行时，奇迹发生了
scores = torch.matmul(q_embed, k_embed.transpose(-2, -1))
```

1. **自动注入：** `q_embed` 携带了绝对位置 $m$ 的旋转，`k_embed` 携带了绝对位置 $n$ 的旋转。
2. **数学魔法：** 当它们做矩阵乘法（点积）时，根据积化和差公式，两者的绝对位置 $m$ 和 $n$ 抵消，留下了 $\cos((m-n)\theta)$。
3. **结果：** `scores` 矩阵中的数值大小，直接受到了 $m-n$ (相对距离) 的控制。
   - 如果距离很近，$\cos$ 值大，Score 趋向保留原始语义相似度。
   - 如果距离很远，不同频率维度的 $\cos$ 值开始震荡抵消，Score 会受到衰减。

这就好比 Q 和 K 都在转动自己的表盘，只有当它们两个对齐（做点积）的那一瞬间，表盘指针的夹角（相对距离）才显现出来，并直接决定了 Attention 分数的高低。

## 6. 深入理解：相干叠加与位置衰减

这里有一个非常关键的误解需要澄清。当我们说"距离为 0 时完全匹配"，实际的求和结果**不是 1，而是 64**（维度对数）。

### 相干叠加（Coherent Superposition）

完整的公式是：

$$S = \sum_{j=0}^{63} \cos((m-n) \cdot \theta_j)$$

假设 head_dim 是 128，那么就有 **64 对** 这样的 $\cos$ 值相加。

#### 情况 A：距离为 0（自己看自己，m=n）

- 距离 $d = 0$
- 对于所有的 $j$，角度都是 $0 \cdot \theta_j = 0$
- $\cos(0) = 1$
- **求和结果：** $1 + 1 + ... + 1$（共 64 个）**= 64**
- **含义：** 这是最强的信号，表示"完全匹配"

#### 情况 B：距离很近（比如 m-n = 1）

- **低频对（转得慢）：** 角度只转了一点点，$\cos \approx 0.999$
- **高频对（转得快）：** 角度转得稍微多点，$\cos \approx 0.9$
- **求和结果：** 大约 $64 \times 0.95 \approx$ **60** 左右
- **含义：** 信号依然很强，只是比"自己看自己"稍弱一点

#### 情况 C：距离很远（比如 m-n = 1000）

- **乱套了：**
  - 有的维度 $\cos$ 是 $0.8$
  - 有的维度 $\cos$ 是 $-0.9$
  - 有的维度 $\cos$ 是 $0.1$
- **求和结果：** 正负相互抵消（Destructive Interference）
- **最终结果：** 可能只有 **2** 或者 **-3**，甚至接近 **0**

### 数值示例

假设 dim=128（64对），base=10000。位置部分的求和结果大致如下：

| 距离 | 求和值 | 归一化后 | 说明 |
|------|--------|----------|------|
| 0 | 64 | 1.0 | 完全匹配，所有 cos 都是 1 |
| 10 | ~51 | ~0.8 | 高频维度开始错位，但低频依然对齐 |
| 100 | ~19 | ~0.3 | 大部分维度已经乱了，只有极低频在贡献 |
| 1000 | ~3 | ~0.05 | 接近噪音水平，几乎无相关性 |

### 这对 Attention 意味着什么？

把这个数值放回 Attention 的公式里：

$$\text{Attention}(Q, K) = \text{Softmax}\left(\frac{\text{Score}}{\sqrt{d}}\right)$$

- 当模型在看**邻居**时，Score 贡献是 **~60**
- 当模型在看**远方**时，Score 贡献是 **~0**

经过 **Softmax**（它是指数放大的）之后：

- $e^{60/\sqrt{d}}$ 是一个很大的数字
- $e^{0}$ 是 1

**结论：** 正是因为这个和**不是恒定的 1**，而是从 64 跌落到 0，才使得 Attention 机制能够"聚焦"。如果不管距离多远和都是 1，那 RoPE 就失效了，模型就变成了"近视眼"，分不清远近。

这个求和结果是一个**反映"位置相似度"的打分**：

- **最高分 = 维度对数（64）** → 距离为 0
- **随着距离增加，分数下降**
- **这就是 RoPE 实现"远程衰减"的物理本质**

## 总结

RoPE 是现代 LLM 的基石之一。它不做加法，而是做旋转；它不存相对位置表，而是通过绝对位置的旋转差值自然导出相对位置。

- **优点 1：** 理论上可以处理任意长度的序列（外推性好）。
- **优点 2：** 计算高效，完美契合 GPU 架构。
- **优点 3：** 显式地将相对距离衰减引入了 Attention 机制（远程衰减）。
