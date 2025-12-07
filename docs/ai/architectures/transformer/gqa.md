---
title: Grouped Query Attention (GQA)
sidebar_label: GQA 分组查询注意力
date: 2025-12-07
last_update:
  date: 2025-12-07
tags: [transformer, attention, gqa, mha, mqa, kv-cache, inference-optimization]
---

# Grouped Query Attention (GQA) 技术文档

## 1. 概述 (Overview)

**Grouped Query Attention (GQA)** 是一种用于 Transformer 模型的高效注意力机制。它由 Google Research 在 2023 年提出（论文：*GQA: Training Generalized Multi-Query Transformer Models*）。

GQA 是 **Multi-Head Attention (MHA)** 和 **Multi-Query Attention (MQA)** 之间的插值方案。旨在解决大语言模型（LLM）在长上下文推理时的显存瓶颈问题，同时保持接近 MHA 的模型性能。

### 核心价值

- **降低显存占用：** 显著减小 KV Cache 的大小（通常减少 4-8 倍）。
- **提升推理速度：** 减少内存带宽压力，大幅提升 Decoding 阶段的吞吐量。
- **保持高精度：** 性能表现优于 MQA，几乎等同于 MHA。

---

## 2. 架构原理 (Architecture)

### 2.1 结构对比

为了理解 GQA，我们需要将其与传统的 MHA 和激进的 MQA 进行对比。假设 Query 头数为 $H$：

| 架构 | Query 头数 | Key/Value 头数 | 比例 (Q:KV) | 特点 |
| :--- | :--- | :--- | :--- | :--- |
| **MHA (Multi-Head)** | $H$ | $H$ | 1 : 1 | 质量最高，显存占用极大。 |
| **GQA (Grouped-Query)** | $H$ | $G$ (其中 $1 < G < H$) | $R$ : 1 | **质量与速度的最佳平衡。** |
| **MQA (Multi-Query)** | $H$ | 1 | $H$ : 1 | 速度最快，但质量有损耗。 |

### 2.2 分组机制

在 GQA 中，我们将 Query 头分成 $G$ 个组，每组包含 $R$ 个 Query 头。

- **组内共享：** 同一组内的 $R$ 个 Query 头共享同一对 Key 和 Value 头。
- **计算逻辑：** 在计算 Attention Score 时，需要将 Key 和 Value 在维度上进行 **广播 (Broadcast)** 或 **复制 (Repeat)**，以匹配 Query 的数量。

**示例配置：**

- Query Heads = 32
- KV Heads = 8
- 分组大小 (Group Size) = 4
- 即：每 4 个 Query 头共用 1 个 KV 头。

---

## 3. 实现细节 (Implementation)

以下是基于 PyTorch 的标准 GQA 模块实现参考。

### 3.1 核心代码逻辑

GQA 的实现与 MHA 非常相似，唯一的区别在于 `forward` 过程中对 KV 张量的处理。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头的数据沿头维度复制，以匹配 Query 的头数。
    input: (Batch, Seq_Len, n_kv_head, Head_Dim)
    output: (Batch, Seq_Len, n_head, Head_Dim)
    """
    if n_rep == 1:
        return x
    B, T, n_kv_head, head_dim = x.shape
    # 使用 repeat_interleave 实现: [k1, k2] -> [k1, k1, k2, k2]
    return x.repeat_interleave(n_rep, dim=2)

class GQA(nn.Module):
    def __init__(self, dim, n_head, n_kv_head):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = dim // n_head

        # 维度检查
        assert n_head % n_kv_head == 0, "Query heads must be divisible by KV heads"

        self.wq = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_head * self.head_dim, bias=False) # 参数量减少
        self.wv = nn.Linear(dim, n_kv_head * self.head_dim, bias=False) # 参数量减少
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # 1. 投影
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        # 2. 这里的 K, V 需要重复以匹配 Q
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 3. 后续计算与标准 Attention 一致
        # Transpose for attention: (B, n_head, T, head_dim)
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        # Attention calculation...
        # ...

        return output
```

---

## 4. 性能分析 (Performance Analysis)

### 4.1 显存占用 (Memory Footprint)

GQA 的主要优势在于对 **KV Cache** 的压缩。

假设模型层数为 $L$，隐藏层维度为 $D$，序列长度为 $S$，Batch大小为 $B$。

KV Cache 的显存占用公式大致为：

$$
\text{Size} = 2 \times B \times S \times L \times D \times \frac{N_{kv}}{N_{head}} \times \text{Precision}
$$

- 如果使用 **MHA** ($N_{kv} = N_{head}$)，系数为 1。
- 如果使用 **GQA** ($N_{kv} = \frac{1}{8} N_{head}$)，系数为 1/8。
- **结论：** GQA 能让同样的显卡支持 **8倍** 的 Batch Size 或 **8倍** 的 Context Length。

### 4.2 精度影响 (Accuracy)

根据 Llama 2 和其他消融实验的结果：

- **MHA vs GQA:** GQA 的 Perplexity (困惑度) 仅比 MHA 增加微不足道的数值（例如 +0.01），在下游任务（Summarization, QA）中表现几乎一致。
- **GQA vs MQA:** GQA 在多步推理和复杂逻辑任务上显著优于 MQA。

---

## 5. 最佳实践与配置 (Best Practices)

### 5.1 参数选择

在设计新模型或进行 Uptraining 时，推荐以下配置：

- **Group Size (分组大小):** 通常建议设置为 **8** (即 `n_head` 是 `n_kv_head` 的 8 倍)。这是目前业界（如 Llama 3）公认的性价比甜点。
- **Uptraining:** 如果从现有的 MHA 模型转换，需要进行约原始预训练步数 5% 的继续训练（Uptraining），以使模型适应共享 KV 的模式。

### 5.2 常见开源模型案例

目前主流模型均已采用 GQA：

- **Llama 2 (70B):** GQA
- **Llama 3 (8B & 70B):** GQA
- **Mistral 7B:** GQA
- **Qwen-2:** GQA

---

## 6. 常见问题 (FAQ)

**Q: 使用 GQA 会导致首字延迟（Time to First Token）变慢吗？**

A: 不会。首字生成主要受限于矩阵乘法计算量（Compute Bound），GQA 的计算量与 MHA 相比仅略微减少（因为 Projection 层变小），所以首字延迟几乎持平。GQA 提升的是后续生成的吞吐量（Memory Bound）。

**Q: 我可以直接修改推理代码来把 MHA 变成 GQA 吗？**

A: 不可以。模型的权重（Weights）必须在训练阶段就按照 GQA 的结构进行训练。你不能直接把训练好的 MHA 权重的 KV 头平均化，那样会导致模型输出乱码。

---

## 7. 参考资料

- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - GQA 原始论文
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Llama 2 技术报告
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - MQA 原始论文
