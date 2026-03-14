---
title: 主流 Transformer 架构详解：GPT、BERT、T5
sidebar_label: 主流 Transformer 架构详解：GPT、BERT、T5
date: 2025-09-28
last_update:
  date: 2025-09-28
---

# 主流 Transformer 架构详解：GPT、BERT、T5

## 一、架构概述

### 1.1 三种架构的本质区别

**GPT（Generative Pre-trained Transformer）** 采用 **Decoder-only** 架构，本质上是一个自回归语言模型。它只使用 Transformer 的解码器部分，通过因果注意力机制（Causal Attention）确保每个位置只能看到之前的信息，这种单向信息流使其天然适合文本生成任务。GPT 的核心思想是"根据前文预测下一个词"，这种训练方式让模型学会了语言的概率分布。

**BERT（Bidirectional Encoder Representations from Transformers）** 采用 **Encoder-only** 架构，是一个双向语言理解模型。它使用完整的注意力机制，允许每个位置同时看到前后文信息。这种双向信息流让 BERT 能够充分理解上下文，但也使其失去了自然的生成能力。BERT 通过掩码语言建模（MLM）进行训练，随机遮盖部分词汇并预测它们，从而学习深层的语言表示。

**T5（Text-to-Text Transfer Transformer）** 采用完整的 **Encoder-Decoder** 架构，将所有 NLP 任务统一为"文本到文本"的转换问题。编码器负责理解输入（使用双向注意力），解码器负责生成输出（使用因果注意力），两者通过交叉注意力机制连接。这种架构既保留了 BERT 的双向理解能力，又具备 GPT 的生成能力，特别适合需要先理解后生成的任务。

### 1.2 注意力机制的关键差异

三种架构最核心的区别在于注意力掩码（Attention Mask）的实现方式：

- **GPT 的因果掩码**：使用下三角矩阵，确保位置 i 只能关注位置 0 到 i 的信息，实现了严格的从左到右的信息流。这种掩码在训练时就内置，使得模型天然学会了自回归生成。

- **BERT 的全注意力**：没有方向性限制，每个位置可以自由地关注序列中的任何其他位置。这种全连接的注意力模式让 BERT 能够建立更丰富的上下文表示，但也意味着它不能直接用于生成任务。

- **T5 的混合模式**：编码器使用全注意力理解输入，解码器使用因果注意力生成输出，同时通过交叉注意力让解码器能够关注编码器的输出。这种设计巧妙地结合了理解和生成的需求。

## 二、核心架构实现

### 2.1 GPT 架构详解与实现

GPT 的架构设计遵循"简单即是美"的原则。整个模型由相同的 Transformer 块堆叠而成，每个块包含一个多头自注意力层和一个前馈网络层，使用残差连接和层归一化来保证深层网络的稳定训练。

GPT 的一个重要设计选择是 **Pre-LayerNorm**，即在子层之前而不是之后进行归一化。这种设计在深层网络中表现出更好的训练稳定性，已经成为现代大语言模型的标准配置。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTBlock(nn.Module):
    """
    GPT的核心构建块
    采用Pre-LN结构：LN → Attention → Residual → LN → FFN → Residual
    这种结构相比Post-LN有更好的训练稳定性
    """
    
    def __init__(self, config):
        super().__init__()
        # Pre-LayerNorm：在attention和FFN之前进行归一化
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # FFN使用4倍的隐藏层维度，这是一个经验性的选择
        # 现代模型如LLaMA使用SwiGLU激活函数，维度比例调整为8/3
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # GPT-2使用GELU，GPT-3后续版本使用GeLU的近似版本
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        # 残差连接确保梯度可以直接流向底层
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制
    核心是因果掩码，确保每个位置只能看到它之前的位置
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # QKV三个投影矩阵合并为一个，提高计算效率
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 因果掩码：下三角矩阵，register_buffer确保它不被当作模型参数
        self.register_buffer("bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 一次性计算Q、K、V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 重塑为多头格式：(B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数，使用缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 应用因果掩码：未来位置设为-inf，softmax后变为0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 应用注意力权重到values
        y = att @ v
        
        # 恢复原始形状
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y
```

### 2.2 BERT 架构详解与实现

BERT 的设计理念是"深度双向理解"。与 GPT 不同，BERT 使用 **Post-LayerNorm** 结构，这是原始 Transformer 的设计。BERT 还引入了三种嵌入：词嵌入、位置嵌入和段落嵌入（用于区分不同的句子），这让它能够处理句对任务。

BERT 最大的创新是通过 MLM（掩码语言模型）任务进行预训练，这种方式让模型能够利用双向上下文，但代价是不能直接用于生成任务。

```python
class BERTBlock(nn.Module):
    """
    BERT的Transformer编码器块
    采用Post-LN结构：Attention → Residual → LN → FFN → Residual → LN
    这是原始Transformer的结构，训练相对不如Pre-LN稳定
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # BERT的FFN使用GELU激活函数，这在当时是一个创新
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力 + 残差连接
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.ln1(hidden_states + attention_output)
        
        # FFN + 残差连接
        intermediate_output = self.activation(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.ln2(hidden_states + layer_output)
        
        return hidden_states

class MultiHeadSelfAttention(nn.Module):
    """
    BERT的多头自注意力
    与GPT的主要区别是没有因果掩码，可以看到完整的序列
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 分离的Q、K、V投影（与GPT的合并方式不同）
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算Q、K、V并重塑为多头
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 缩放点积注意力
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力掩码（主要用于padding）
        if attention_mask is not None:
            # attention_mask是1/0矩阵，1表示真实token，0表示padding
            extended_attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # 注意：这里没有因果掩码，每个位置可以看到所有位置
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # 输出投影
        output = self.dense(context_layer)
        return output
    
    def transpose_for_scores(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
```

### 2.3 T5 架构详解与实现

T5 的设计哲学是"统一即是力量"。它将所有 NLP 任务都转换为文本到文本的格式，使用相同的模型、损失函数和超参数。T5 使用了相对位置编码而不是绝对位置编码，这让它能够更好地泛化到不同长度的序列。

T5 的编码器-解码器架构需要处理三种注意力：编码器自注意力（双向）、解码器自注意力（因果）和解码器-编码器交叉注意力（让解码器关注编码器的输出）。

```python
class T5Block(nn.Module):
    """
    T5的Transformer块，可配置为编码器或解码器
    T5使用简化的层归一化（RMSNorm）和相对位置编码
    """
    
    def __init__(self, config, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        
        # T5使用Pre-LN，但使用简化版的RMSNorm而不是标准LayerNorm
        self.layer_norm_1 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # 自注意力层
        self.self_attention = T5Attention(
            config,
            is_decoder=is_decoder,
            use_bias=False,  # T5不使用bias以减少参数
            relative_attention_bias=True  # 使用相对位置编码
        )
        self.dropout_1 = nn.Dropout(config.dropout_rate)
        
        # 交叉注意力层（仅解码器使用）
        if is_decoder:
            self.layer_norm_2 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            self.cross_attention = T5Attention(
                config,
                is_decoder=False,  # 交叉注意力不需要因果掩码
                use_bias=False,
                relative_attention_bias=False  # 交叉注意力不需要相对位置
            )
            self.dropout_2 = nn.Dropout(config.dropout_rate)
        
        # FFN层
        self.layer_norm_3 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ffn = T5FFN(config)
        self.dropout_3 = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # 自注意力
        normed_hidden_states = self.layer_norm_1(hidden_states)
        attention_output = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout_1(attention_output)
        
        # 交叉注意力（仅解码器）
        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer_norm_2(hidden_states)
            cross_attention_output = self.cross_attention(
                normed_hidden_states,
                key_value_states=encoder_hidden_states,
                mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + self.dropout_2(cross_attention_output)
        
        # FFN
        normed_hidden_states = self.layer_norm_3(hidden_states)
        ffn_output = self.ffn(normed_hidden_states)
        hidden_states = hidden_states + self.dropout_3(ffn_output)
        
        return hidden_states

class T5Attention(nn.Module):
    """
    T5的注意力机制
    支持编码器（双向）、解码器（因果）和交叉注意力
    使用相对位置编码而不是绝对位置编码
    """
    
    def __init__(self, config, is_decoder=False, use_bias=False, relative_attention_bias=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.relative_attention_bias = relative_attention_bias
        self.d_model = config.d_model
        self.n_heads = config.num_heads
        self.d_kv = config.d_kv  # T5允许K、V的维度与Q不同，以节省参数
        
        # Q、K、V投影
        self.q = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.k = nn.Linear(self.d_model, self.d_kv * self.n_heads, bias=use_bias)
        self.v = nn.Linear(self.d_model, self.d_kv * self.n_heads, bias=use_bias)
        self.o = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        
        # 相对位置编码
        if self.relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(32, self.n_heads)
    
    def forward(self, hidden_states, key_value_states=None, mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 如果提供了key_value_states，说明是交叉注意力
        if key_value_states is None:
            key_value_states = hidden_states
        
        # 计算Q、K、V
        q = self.q(hidden_states).view(batch_size, seq_length, self.n_heads, self.d_model // self.n_heads)
        k = self.k(key_value_states).view(batch_size, -1, self.n_heads, self.d_kv)
        v = self.v(key_value_states).view(batch_size, -1, self.n_heads, self.d_kv)
        
        # 转置为(batch, heads, seq_len, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_kv)
        
        # 添加相对位置偏置
        if self.relative_attention_bias:
            position_bias = self.compute_position_bias(seq_length)
            scores = scores + position_bias
        
        # 应用掩码（因果掩码或padding掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 如果是解码器，应用因果掩码
        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(seq_length, seq_length)).to(scores.device)
            scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        # 输出投影
        attn_output = self.o(attn_output)
        return attn_output
    
    def compute_position_bias(self, length):
        """计算相对位置偏置"""
        # T5使用学习的相对位置编码
        # 位置差异被分桶，近距离有更细粒度的区分
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values
```

## 三、架构特性对比

### 3.1 计算复杂度分析

| 架构特性 | GPT | BERT | T5 |
|----------|-----|------|-----|
| **时间复杂度** | O(n²) | O(n²) | O(n²+nm) |
| **空间复杂度** | O(n²) | O(n²) | O(n²+nm) |
| **并行度** | 低（生成时串行） | 高（完全并行） | 中等 |
| **KV缓存优化** | 支持 | 不需要 | 仅解码器支持 |

其中 n 是序列长度，m 是编码器序列长度（用于 T5 的交叉注意力）。

### 3.2 架构设计权衡

**GPT 的权衡**：
- 优势：训练和推理一致性好，扩展性优秀，工程实现简单
- 劣势：不能利用后文信息，生成速度受自回归限制
- 设计哲学：简单统一，通过规模弥补架构限制

**BERT 的权衡**：
- 优势：双向理解能力强，特征提取效果好，推理速度快
- 劣势：不能直接生成，存在预训练-微调差异，扩展性受限
- 设计哲学：深度理解，专注于编码器任务

**T5 的权衡**：
- 优势：任务统一，既能理解又能生成，多任务学习效果好
- 劣势：参数量大（双栈结构），训练成本高，推理复杂
- 设计哲学：统一框架，一个模型解决所有任务

## 四、现代发展趋势

### 4.1 为什么 GPT 成为主流

1. **扩展定律（Scaling Laws）验证**：GPT 架构在参数量增加时性能提升最稳定
2. **工程友好**：单栈结构简单，优化技术成熟（KV cache、Flash Attention等）
3. **生成能力需求**：当前应用场景（对话、代码生成）都需要强生成能力
4. **训练效率**：相比 T5 的双栈结构，单栈的 GPT 训练效率更高

### 4.2 架构创新方向

- **长上下文处理**：RoPE、ALiBi 等位置编码创新支持更长序列
- **稀疏注意力**：Flash Attention、Sparse Transformer 降低复杂度
- **混合专家（MoE）**：Mixtral 等模型通过稀疏激活提升容量
- **架构搜索**：自动发现更优的 Transformer 变体

## 参考资料

- [Attention Is All You Need - Transformer 原论文](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners - GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Transformer 架构演进综述 2024](https://arxiv.org/abs/2304.13712)