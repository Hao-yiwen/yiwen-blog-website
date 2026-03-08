---
title: MoE架构详解与代码实现
sidebar_label: MoE架构详解
date: 2025-12-07
tags: [MoE, Mixture of Experts, DeepSeek, LLM, 稀疏激活]
---

# MoE 架构详解与代码实现

本文档详细介绍 **标准 MoE (Standard MoE)** 和 **共享+路由 MoE (Shared + Routed MoE)** 两种架构。

## 0. 背景与动机 (Background)

### 为什么需要 MoE？
随着大语言模型 (LLM) 的发展，Scaling Law 告诉我们：**参数量越大，模型性能越强**。
但是，传统的**稀密模型 (Dense Model)** 面临着一个巨大的瓶颈：
*   **计算量与参数量绑定**：在 Dense 模型中，处理每一个 Token 都需要激活模型的所有参数。如果我们想把模型参数扩大 10 倍，那么训练和推理的计算成本（FLOPs）也会增加 10 倍。
*   **训练成本墙**：为了追求更强的智能，我们需要万亿甚至更大参数的模型，但现有的硬件算力难以支撑与之成正比的巨额计算量。

**Mixture of Experts (MoE)** 的出现就是为了打破这个"铁律"。

### MoE 的核心价值
*   **稀疏激活 (Sparse Activation)**：MoE 允许模型拥有极大的参数量（例如 1000B+），但在处理单个 Token 时，只激活其中极小的一部分（例如 10B）。
*   **解耦计算与参数**：这意味着我们可以享受到**超大模型带来的知识容量**，同时只需要支付**小模型的计算成本**。

### 在 Transformer 中的位置
在标准的 Transformer 架构中，计算量最大的部分通常是 **Feed-Forward Network (FFN)** 层（参数量约占整体的 2/3）。
因此，MoE 最经典的做法就是**将 Transformer Block 中的 FFN 层替换为 MoE 层**。
*   **Dense Block**: `Self-Attention -> Add & Norm -> Dense FFN -> Add & Norm`
*   **MoE Block**: `Self-Attention -> Add & Norm -> MoE Layer (Experts) -> Add & Norm`

---

## 1. 标准 MoE (Standard MoE)

### 核心机制
标准 MoE 包含一组专家网络（通常是结构相同的 MLP）和一个门控网络（Router）。
1.  **Router**: 接收输入，输出从 N 个专家中选择 k 个专家的概率。
2.  **Top-k**: 仅计算这 k 个专家的输出。
3.  **加权和**: 最终输出是这 k 个专家输出的加权总和。

### 代码实现 (PyTorch 演示版)
*注：为了便于理解原理，下方代码使用了 Python 循环。在实际的大规模训练中，通常会使用 `torch.gather` 可以在 GPU 上并行的高效实现。*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k, d_ff):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 门控网络 (Router)
        # 简单的线性层，将维度映射到专家数量
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # 2. 专家网络 (Experts)
        # 包含 num_experts 个独立的 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]

        # --- Router 决策 ---
        router_logits = self.router(x)  # [B, T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # 选出概率最高的 top_k 个专家
        # weights: 对应的概率值, indices: 专家的ID
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)

        # 归一化权重 (让选中的权重之和为1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x)

        # --- 专家计算 (逻辑演示) ---
        # 展平以便处理
        flat_x = x.view(-1, x.size(-1))
        flat_out = final_output.view(-1, x.size(-1))

        # 遍历每一个"名次" (top-1 到 top-k)
        for k in range(self.top_k):
            # 取出当前名次的专家索引和权重
            expert_idx = indices[:, :, k].view(-1)      # [B*T]
            expert_w = weights[:, :, k].view(-1, 1)     # [B*T, 1]

            # 针对每一个具体的专家进行计算
            for e_id in range(self.num_experts):
                # 找出分配给专家 e_id 的所有 token
                mask = (expert_idx == e_id)
                if mask.sum() > 0:
                    # 1. 选出 Token
                    tokens = flat_x[mask]
                    # 2. 专家前向传播
                    out = self.experts[e_id](tokens)
                    # 3. 加权累加回结果
                    flat_out[mask] += out * expert_w[mask]

        return final_output
```

---

## 2. 共享 + 路由 MoE (Shared + Routed MoE)

### 核心机制
这种架构（如 DeepSeek-V2/V3 采用的设计）将参数分为两部分：
1.  **共享专家 (Shared Expert)**: **总是被激活**。负责捕获通用的、共性的知识（Common Knowledge）。
2.  **路由专家 (Routed Experts)**: **按需稀疏激活**。负责特定的、细分的知识。

**最终输出** = `共享专家输出` + `Router选中的路由专家加权输出`。

### 代码实现 (PyTorch 演示版)

```python
class SharedRoutedMoE(nn.Module):
    def __init__(self, d_model, num_routed_experts, top_k, d_ff, d_shared_ff):
        super().__init__()
        self.top_k = top_k
        self.num_routed_experts = num_routed_experts

        # 1. 共享专家 (Shared Experts)
        # 不受 Router 控制，总是处理所有 Token
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, d_shared_ff),
            nn.ReLU(),
            nn.Linear(d_shared_ff, d_model)
        )

        # 2. 路由专家 (Routed Experts)
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_routed_experts)
        ])

    def forward(self, x):
        # x shape: [B, T, d_model]

        # --- 路径 A: 共享专家 (Shared Path) ---
        # 就像一个普通的 FFN，直接计算
        shared_output = self.shared_expert(x)

        # --- 路径 B: 路由专家 (Routed Path) ---
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(router_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        routed_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_routed_out = routed_output.view(-1, x.size(-1))

        # 路由计算逻辑 (同 Standard MoE)
        for k in range(self.top_k):
            expert_idx = indices[:, :, k].view(-1)
            expert_w = weights[:, :, k].view(-1, 1)

            for e_id in range(self.num_routed_experts):
                mask = (expert_idx == e_id)
                if mask.sum() > 0:
                    tokens = flat_x[mask]
                    out = self.routed_experts[e_id](tokens)
                    flat_routed_out[mask] += out * expert_w[mask]

        # --- 最终融合 ---
        # 将共享路径和路由路径的结果相加
        final_output = shared_output + routed_output

        return final_output
```

### 两种架构对比总结

| 维度 | 标准 MoE | 共享 + 路由 MoE |
| :--- | :--- | :--- |
| **FFN 替代方式** | 完全由稀疏专家替代 | 稠密 FFN (共享) + 稀疏 FFN (路由) 混合 |
| **知识分工** | 每个专家都混合学习通用和特定知识 | **共享专家**学通用，**路由专家**学特定 |
| **参数利用率** | 可能存在冗余（多个专家学重复知识） | 更高，减少了对通用知识的重复存储 |
