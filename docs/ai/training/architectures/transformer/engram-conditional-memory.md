---
title: DeepSeek Engram 条件记忆架构详解
sidebar_label: Engram 条件记忆
sidebar_position: 25
date: 2025-01-16
tags: [deepseek, engram, n-gram, moe, conditional-memory, llm-architecture, sparsity]
---

# DeepSeek Engram：大型语言模型的"条件记忆"新维度

本文是对 DeepSeek-AI 论文 **《Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models》** 的深度解读。

Engram 是对经典 N-gram 模型的现代化改造，通过引入"条件记忆"维度，与 MoE（混合专家）形成完美互补，显著提升了大型语言模型的效率和性能。

---

## 1. 核心理念与动机

### 1.1 问题现状：用"算力"模拟"记忆"的低效

当前的 Transformer 模型（包括 MoE）缺乏原生的"查表"机制。当模型需要处理诸如 "Alexander the Great"（亚历山大大帝）这类固定知识时，必须通过消耗深层网络的注意力机制和前馈网络（FFN）来逐步"拼凑"和重构这些概念。

**问题：**
- 浪费了宝贵的计算深度
- 导致用于处理复杂逻辑推理的资源被挤占
- 固定知识的处理效率低下

### 1.2 解决方案：计算与记忆的解耦

论文提出将模型能力拆解为两个正交但互补的**稀疏性维度**：

| 维度 | 承担者 | 职责 | 复杂度 |
|------|--------|------|--------|
| **条件计算** | MoE (Mixture-of-Experts) | 动态逻辑、推理、复杂上下文依赖 | 高 |
| **条件记忆** | Engram | 静态世界知识（百科、短语、成语等） | $O(1)$ |

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer 模型                      │
│  ┌─────────────────┐         ┌─────────────────┐        │
│  │     MoE 专家     │  互补   │     Engram      │        │
│  │   条件计算       │◄──────►│    条件记忆      │        │
│  │  (动态推理)      │         │   (静态查表)    │        │
│  └─────────────────┘         └─────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Engram 技术架构详解

Engram 作为一个**即插即用的模块**集成在 Transformer 骨干网络中。

### 2.1 预处理与检索

#### Tokenizer 压缩

为了解决现代分词器的冗余（如 "Apple" vs "apple"），Engram 首先将语义等价的 Token 映射为统一的 ID。

- **压缩率：** 约 23%
- **效果：** 提高了语义密度，减少了冗余查询

```python
# 概念示例
token_map = {
    "Apple": 1001,
    "apple": 1001,  # 映射到同一ID
    "APPLE": 1001,
}
```

#### 混合 N-gram (Mixed N-gram)

模型**同时**提取当前位置的后缀 **2-gram** 和 **3-gram**：

```
输入: "I love machine learning"

位置 "learning" 的 N-gram:
  2-gram: ("machine", "learning")
  3-gram: ("love", "machine", "learning")
```

#### 多头哈希 (Multi-Head Hashing)

为了在有限空间内存储海量组合，使用多个哈希函数将 N-gram 映射到嵌入表索引。

$$
\text{index}_h = \text{Hash}_h(n\text{-}gram) \mod M
$$

其中 $h$ 是哈希头索引，$M$ 是表大小。

**优势：**
- 确定性的映射
- 允许模型扩展到 100B+ 参数
- 不占用过多显存

### 2.2 上下文融合

单纯查出来的静态向量可能包含噪声（哈希冲突）或歧义。Engram 引入了**动态门控机制**。

#### 上下文感知门控 (Context-Aware Gating)

使用当前 Transformer 主干流的隐藏状态 $h$ 作为 Query，查出来的记忆向量 $m$ 作为 Key/Value。

计算门控系数 $g$：

$$
g = \sigma(W_g \cdot [h; m])
$$

其中 $\sigma$ 是 sigmoid 函数，$W_g$ 是可学习参数。

**关键特性：** 如果当前上下文判定查出来的知识不相关，$g$ 会趋近于 0，从而**抑制噪声**。

#### 残差连接 (Residual Connection)

最终的输出通过残差加法融入主干流：

$$
h' = h + g \odot m
$$

这意味着 Engram 是对现有信息的"**增强**"而非"替换"。

### 2.3 完整数据流

```
输入 Tokens
     │
     ▼
┌────────────────┐
│ Tokenizer 压缩 │  (语义归一化)
└────────────────┘
     │
     ▼
┌────────────────┐
│ 提取 2-gram    │
│ 提取 3-gram    │
└────────────────┘
     │
     ▼
┌────────────────┐
│ 多头哈希映射    │  ──► 嵌入表查询 ──► 记忆向量 m
└────────────────┘
     │
     ▼
┌────────────────┐
│ 上下文门控融合  │  h' = h + g ⊙ m
└────────────────┘
     │
     ▼
输出到下一层
```

### 2.4 部署策略

#### 层数分布

Engram 并不每层都加，通常是**稀疏插入**。

例如在 27B 模型中：
- **第 2 层**：早期预热/卸载（提前注入知识）
- **第 15 层**：利用深层上下文消除歧义

#### 参数独立

不同层的 Engram 模块拥有**独立的参数和哈希表**，各司其职。

---

## 3. 稀疏性分配定律

论文通过实验揭示了 MoE 和 Engram 之间的最佳资源配比关系。

### 3.1 U 型缩放定律

在固定总参数量和计算量（Iso-FLOPs）的前提下，性能与 Engram 的参数占比呈 **U 型关系**：

```
Loss ▲
     │    ╲                    ╱
     │     ╲                  ╱
     │      ╲                ╱
     │       ╲──────────────╱
     │              ▲
     │          最优点 (20-25%)
     └──────────────────────────────► Engram 参数占比
         0%        25%       50%      100%
```

### 3.2 黄金比例

| 组件 | 参数预算占比 | 职责 |
|------|-------------|------|
| **Engram** | 20% ~ 25% | 条件记忆 |
| **MoE 专家** | 75% ~ 80% | 条件计算 |

这一比例能获得最佳的模型性能（Loss 最低）。

### 3.3 无限内存扩展

如果不限制总参数量（只限制计算量），单纯增加 Engram 的插槽数量（Memory Slots），模型性能会呈**对数线性持续提升**：

$$
\text{Performance} \propto \log(\text{Memory Slots})
$$

---

## 4. 系统级优化与效率

Engram 的设计充分利用了现代硬件架构（CPU/GPU 异构），解决了显存瓶颈。

### 4.1 确定性寻址

与 MoE 需要计算后才知道路由给哪个专家不同，Engram 的查表 ID **仅依赖于输入文本 (Input Tokens)**。

```
时间线：
─────────────────────────────────────────────────────────►
│        │        │        │        │
│ 输入   │ 第1层  │ 第2层  │ 第3层  │
│ 解析   │ 计算   │ 计算   │ 计算   │
│        │        │        │        │
│        │        │        │        │
│ ◄───── CPU 预取第2层 Engram 数据 ─────►
│        │        │        │        │
└────────┴────────┴────────┴────────┘
```

**关键洞察：** 在 GPU 开始计算第 1 层之前，CPU 就已经知道第 2 层需要查哪些表了。

### 4.2 预取与卸载 (Prefetch & Offloading)

利用上述特性，可以实现高效的内存管理：

| 存储位置 | 内容 | 优势 |
|----------|------|------|
| **GPU 显存** | Transformer 权重、激活 | 高带宽计算 |
| **CPU 内存 (Host DRAM)** | Engram 嵌入表 (100B+) | 大容量、低成本 |

**工作流程：**
1. 解析输入文本，确定需要的 N-gram
2. 在 GPU 计算前序层时，通过 PCIe 异步预取数据
3. 数据到达时正好用于当前层计算

**结果：** 即使外挂 100B 参数，推理延迟损耗也 **< 3%**。

### 4.3 性能对比

| 配置 | 显存占用 | 推理延迟 | 参数规模 |
|------|----------|----------|----------|
| 纯 GPU | 高 | 基准 | 受限于显存 |
| Engram + 卸载 | 低 | +2.8% | 100B+ |

---

## 5. 实验结论与影响

### 5.1 性能提升

在同等参数和计算量下，Engram-27B 全面优于 MoE-27B：

| 任务类型 | 基准测试 | 提升幅度 | 说明 |
|----------|----------|----------|------|
| **知识类** | MMLU | +3.4 | 符合预期 |
| **知识类** | CMMLU | +4.0 | 中文知识 |
| **推理** | BBH | +5.0 | **意外之喜** |
| **数学** | MATH | +2.4 | 推理释放 |
| **代码** | HumanEval | +3.0 | 编程能力 |

**关键发现：** 推理和代码能力的提升证明了**卸载记忆负担后，模型能更好地进行逻辑推理**。

### 5.2 机制分析

#### 增加有效深度

CKA (Centered Kernel Alignment) 分析显示：

```
表征能力等效关系：

Engram 模型第 5 层 ≈ 基线模型第 12 层
```

Engram 让模型"**赢在起跑线**"，浅层就具备了深层的表征能力。

#### 释放注意力

由于局部短语被 Engram 处理了，Transformer 的注意力机制可以专注于**全局上下文**：

| 能力 | 基线模型 | Engram 模型 | 原因 |
|------|----------|-------------|------|
| 长文本检索 (NIAH) | 一般 | **大幅提升** | 注意力资源释放 |
| 局部短语理解 | 占用注意力 | Engram 处理 | 分工明确 |

---

## 6. 代码概念示例

### 6.1 Engram 模块伪代码

```python
import torch
import torch.nn as nn

class EngramModule(nn.Module):
    def __init__(self, hidden_size, num_slots, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # 多头嵌入表
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(num_slots, hidden_size)
            for _ in range(num_heads)
        ])

        # 门控网络
        self.gate_proj = nn.Linear(hidden_size * 2, 1)

        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def hash_ngram(self, ngram_ids, head_idx, num_slots):
        """多头哈希函数"""
        # 简化的哈希实现
        prime = 31 + head_idx * 7
        hash_val = 0
        for token_id in ngram_ids:
            hash_val = (hash_val * prime + token_id) % num_slots
        return hash_val

    def forward(self, hidden_states, ngram_indices):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] - Transformer 隐藏状态
            ngram_indices: List of n-gram token ID tuples
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 多头查表并聚合
        memory_vectors = []
        for head_idx, table in enumerate(self.embedding_tables):
            # 计算哈希索引
            indices = [
                self.hash_ngram(ng, head_idx, table.num_embeddings)
                for ng in ngram_indices
            ]
            indices = torch.tensor(indices, device=hidden_states.device)

            # 查表
            mem = table(indices)  # [seq_len, hidden_size]
            memory_vectors.append(mem)

        # 多头平均
        memory = torch.stack(memory_vectors).mean(dim=0)  # [seq_len, hidden_size]
        memory = memory.unsqueeze(0).expand(batch_size, -1, -1)

        # 上下文感知门控
        concat = torch.cat([hidden_states, memory], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))  # [batch, seq_len, 1]

        # 残差融合
        output = hidden_states + gate * self.output_proj(memory)

        return output


# 使用示例
engram = EngramModule(
    hidden_size=4096,
    num_slots=10_000_000,  # 1000万插槽
    num_heads=4
)

# 模拟输入
hidden = torch.randn(2, 128, 4096)  # batch=2, seq=128
ngrams = [(101, 202), (202, 303), ...]  # N-gram token IDs

output = engram(hidden, ngrams)
```

### 6.2 N-gram 提取

```python
def extract_ngrams(token_ids, n_values=[2, 3]):
    """
    提取混合 N-gram

    Args:
        token_ids: Token ID 序列
        n_values: 要提取的 N-gram 大小列表

    Returns:
        每个位置的 N-gram 列表
    """
    ngrams_per_position = []

    for i in range(len(token_ids)):
        position_ngrams = []
        for n in n_values:
            if i >= n - 1:
                ngram = tuple(token_ids[i - n + 1 : i + 1])
                position_ngrams.append(ngram)
        ngrams_per_position.append(position_ngrams)

    return ngrams_per_position


# 示例
tokens = [101, 202, 303, 404, 505]  # "I love machine learning model"
ngrams = extract_ngrams(tokens, n_values=[2, 3])

# 输出:
# 位置 0: []
# 位置 1: [(101, 202)]
# 位置 2: [(202, 303), (101, 202, 303)]
# 位置 3: [(303, 404), (202, 303, 404)]
# 位置 4: [(404, 505), (303, 404, 505)]
```

---

## 7. 与相关技术的对比

| 技术 | 类型 | 复杂度 | 记忆能力 | 适用场景 |
|------|------|--------|----------|----------|
| **N-gram** | 统计模型 | $O(1)$ | 极短 (N-1) | 传统 NLP |
| **Transformer** | 神经网络 | $O(n^2)$ | 全局 | 现代 LLM |
| **MoE** | 稀疏计算 | $O(k \cdot d)$ | 依赖 Transformer | 大规模模型 |
| **Engram** | 条件记忆 | $O(1)$ | 局部短语 | MoE 增强 |
| **RAG** | 检索增强 | $O(\log n)$ | 外部知识库 | 知识密集任务 |

### Engram vs RAG

| 维度 | Engram | RAG |
|------|--------|-----|
| 检索粒度 | Token/短语级 | 文档/段落级 |
| 延迟 | 极低 ($O(1)$) | 较高 (需要检索) |
| 知识更新 | 需重训练 | 可动态更新 |
| 集成方式 | 模型内部 | 模型外部 |

---

## 8. 总结

Engram 不是对 Transformer 的颠覆，而是对其"记忆短板"的**精准补全**。

### 核心贡献

1. **理论创新**：提出计算与记忆的正交分解，定义了"条件记忆"新维度
2. **架构设计**：混合 N-gram + 多头哈希 + 上下文门控的优雅组合
3. **系统优化**：CPU/GPU 异构预取，100B 参数仅增加 3% 延迟
4. **实证验证**：知识、推理、代码任务全面提升

### 关键洞察

> **"查表"** 这种古老的技术在 LLM 时代依然具有巨大的价值。

Engram 将模型从"死记硬背"中解放出来，使其成为了 **MoE 架构的最佳拍档**。

### 未来展望

- 更大规模的记忆表（1T+ 参数）
- 动态知识更新机制
- 与 RAG 的混合架构

---

## 参考资料

- [DeepSeek-AI: Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2501.xxxxx)
- [N-gram 语言模型详解](/docs/ai/nlp/n-gram)
- [马尔可夫链详解](/docs/ai/nlp/markov-chain)
- [DeepSeek V3 架构分析](/docs/ai/architectures/transformer/deepseek-v3-architecture)
