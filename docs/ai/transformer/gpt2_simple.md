---
title: "GPT-2 极简实现"
sidebar_label: "GPT-2 极简实现"
date: 2025-10-09
last_update:
  date: 2025-10-09
---

# GPT-2 极简实现

## 概述

本文档提供了一个可读性优先的 GPT-2 完整实现,旨在帮助理解 Transformer 语言模型的核心架构和全流程工作原理。该实现包含以下关键组件:

- **绝对位置嵌入(Absolute Position Embedding)**
- **预归一化(Pre-Normalization)**
- **遮罩多头注意力(Masked Multi-Head Attention)**
- **前馈网络(MLP)**

## 完整代码实现

```py
# 一个可读性优先的 GPT-2 极简实现：绝对位置嵌入 + 预归一化 + 遮罩多头注意力 + MLP
# 仅用于理解全流程与调试；省略了并行/缓存/FlashAttn 等工程优化

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======= 配置 =======
class GPTConfig:
    def __init__(
        self,
        vocab_size: int = 50257,  # 词表大小
        block_size: int = 128,    # 最大序列长度 T
        n_layer: int = 4,         # Transformer 层数
        n_head: int = 8,          # 注意力头数
        n_embd: int = 512,        # 模型维度 d_model
        dropout: float = 0.1,     # dropout 概率
        tie_weights: bool = True, # LM 头与词嵌入权重共享（GPT-2 风格）
        debug: bool = False,      # 调试形状打印
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.debug = debug


# ======= 遮罩多头自注意力 =======
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能整除 n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.debug = config.debug

        # 将输入一次性线性映射为 Q,K,V（维度 3*n_embd），然后再拆分
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # 注意力输出的投影（对应论文里的 W_O）
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # 预生成因果遮罩，下三角为 True（允许看），上三角为 False（屏蔽）
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        # 形状 [1,1,T,T] 以便广播到 [B,h,T,T]
        self.register_buffer("attn_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

        self.attn_drop = nn.Dropout(self.dropout) 
        self.resid_drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, d_model]
        B, T, C = x.shape
        if self.debug:
            print(f"[Attn] x: {x.shape}")

        # 一次性得到 q,k,v: [B, T, 3*d_model]
        qkv = self.qkv(x)
        if self.debug:
            print(f"[Attn] qkv: {qkv.shape}")

        # 切分并重排为多头：q/k/v: [B, n_head, T, head_dim]
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,h,T,d]
        if self.debug:
            print(f"[Attn] q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # 注意力分数: [B,h,T,T]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果遮罩：只允许看当前位置及左侧
        mask = self.attn_mask[:, :, :T, :T]  # 裁剪到当前 T
        att = att.masked_fill(~mask, float("-inf"))

        # softmax -> dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        if self.debug:
            print(f"[Attn] att(sftmx): {att.shape}, sum over last dim≈1 -> {att[0,0,0,:5].sum().item():.3f}")

        # 加权求和拿到输出： [B,h,T,d]
        y = att @ v

        # 合并头： [B,T,h*d]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影 + dropout
        y = self.resid_drop(self.proj(y))
        if self.debug:
            print(f"[Attn] out: {y.shape}")
        return y


# ======= 两层 MLP（GELU） =======
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)  # 扩张
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)  # 投回
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)
        self.debug = config.debug

    def forward(self, x):
        if self.debug:
            print(f"[MLP] in: {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.debug:
            print(f"[MLP] out: {x.shape}")
        return x


# ======= Transformer Block（预归一化） =======
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.debug = config.debug

    def forward(self, x):
        if self.debug:
            print(f"[Block] in: {x.shape}")
        # 注意力子层（Pre-LN）
        x = x + self.attn(self.ln1(x))
        # MLP 子层（Pre-LN）
        x = x + self.mlp(self.ln2(x))
        if self.debug:
            print(f"[Block] out: {x.shape}")
        return x


# ======= GPT 主体 =======
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 词嵌入 & 绝对位置嵌入（GPT-2 风格）
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)       # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)       # position embedding

        self.drop = nn.Dropout(config.dropout)

        # N 层 Block
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 词表投影头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：lm_head.weight 与 wte.weight 绑定（GPT-2 习惯）
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        # 参数初始化（接近 GPT-2 的简单方案）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: [B,T] 的 token id
        targets: [B,T] 的标签（可选；传入则返回交叉熵 loss）
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "序列长度超过了 block_size"

        if self.config.debug:
            print(f"\n=== Forward: idx {idx.shape} ===")

        # 构造位置索引 [0..T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # [1,T]

        # 词嵌入 + 位置嵌入
        tok_emb = self.wte(idx)        # [B,T,d]
        pos_emb = self.wpe(pos)        # [1,T,d]，会广播到 [B,T,d]
        x = self.drop(tok_emb + pos_emb)
        if self.config.debug:
            print(f"[Emb] tok_emb: {tok_emb.shape}, pos_emb: {pos_emb.shape}, x: {x.shape}")

        # 堆叠的 Transformer blocks
        for i, block in enumerate(self.h, start=1):
            if self.config.debug:
                print(f"\n-- Block {i} --")
            x = block(x)

        # 最终 LN + 词表头
        x = self.ln_f(x)               # [B,T,d]
        logits = self.lm_head(x)       # [B,T,vocab]
        if self.config.debug:
            print(f"\n[Head] logits: {logits.shape}")

        # 训练时可直接算 CE loss
        loss = None
        if targets is not None:
            # 把 (B,T,V) 改成 (B*T,V) 与 (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """
        朴素自回归生成（不做 KV Cache，适合小规模演示）
        idx: [B,T] 初始提示
        """
        for _ in range(max_new_tokens):
            # 只保留最近 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # 取最后一个时间步的分布
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 采样一个 token
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ======= 最小可运行示例 =======
if __name__ == "__main__":
    # 小模型 + 小词表，方便 CPU/GPU 都能跑
    cfg = GPTConfig(
        vocab_size=100,   # 玩具词表
        block_size=16,    # 最长 16 token
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.1,
        debug=True,       # 打开调试打印（形状/关键中间量）
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(device)
    model.train()

    # 构造一批假数据：B=2, T=8
    B, T = 2, 8
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # 单次前向 + 计算 loss
    logits, loss = model(x, y)
    print(f"\nLoss: {loss.item():.4f}")

    # 反向与一次优化步（演示）
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 可选：梯度裁剪
    opt.step()
    print("One optimization step done.")

    # 切推理模式，演示生成
    model.eval()
    start = torch.randint(0, cfg.vocab_size, (1, 4), device=device)  # 提示 4 token
    out = model.generate(start, max_new_tokens=8, temperature=1.0, top_k=20)
    print("Generated ids:", out.tolist())
```

## 架构说明

本实现完整复现了 GPT-2 论文的核心架构,主要包含以下模块:

### 1. 嵌入层(Embedding Layer)
- **Token Embedding**: 将词汇表中的 token 映射为向量表示
- **Position Embedding**: 绝对位置编码,为每个位置学习独立的嵌入向量

### 2. Transformer Block
每个 Transformer 层采用**预归一化(Pre-LN)**架构:
- **残差连接(Residual Connection)**: 有效缓解梯度消失问题
- **遮罩多头注意力(Causal Self-Attention)**: 确保自回归特性,只能关注当前位置及之前的内容
- **前馈网络(MLP)**: 两层全连接网络,中间维度为 `4 * n_embd`,使用 GELU 激活函数

### 3. 语言模型头(LM Head)
- 将最终的隐藏状态投影回词表大小的 logits
- 支持**权重共享(Weight Tying)**: LM Head 与 Token Embedding 共享参数,减少模型大小

## 理论背景

本实现基于 OpenAI 的 GPT-2 论文:
[《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

**核心思想**: 通过大规模无监督预训练,语言模型可以学习到丰富的语言知识和推理能力,从而在多种下游任务上展现出 zero-shot 或 few-shot 学习能力,无需针对特定任务进行微调。

## 使用说明

该实现可用于:
- 学习 GPT 架构的内部工作原理
- 理解因果语言模型的训练和推理流程
- 调试和实验小规模 Transformer 模型
- 作为教学演示代码

:::tip
开启 `debug=True` 可以打印每层的 tensor 形状,帮助理解数据流转过程。
:::