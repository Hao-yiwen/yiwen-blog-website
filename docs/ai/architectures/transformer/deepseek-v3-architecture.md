---
title: DeepSeek-V3.2 模型架构详解
sidebar_label: DeepSeek-V3.2 架构详解
tags: [DeepSeek, MoE, MLA, Transformer, LLM, FP8]
---

# DeepSeek-V3.2 模型架构详解

DeepSeek-V3.2 是 DeepSeek 系列的最新模型，引入了许多独特的架构创新，如 **MLA（Multi-Head Latent Attention）** 和 **DeepSeekMoE**。本文将通过分析其 `config.json` 配置文件，详细解释每个参数的含义，并计算模型的总参数量。

## 配置文件概览

```json
{
  "architectures": ["DeepseekV32ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 1,
  "ep_size": 1,
  "first_k_dense_replace": 3,
  "hidden_act": "silu",
  "hidden_size": 7168,
  "index_head_dim": 128,
  "index_n_heads": 64,
  "index_topk": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 18432,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v32",
  "moe_intermediate_size": 2048,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 256,
  "n_shared_experts": 1,
  "norm_topk_prob": true,
  "num_attention_heads": 128,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 61,
  "num_key_value_heads": 128,
  "num_nextn_predict_layers": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128]
  },
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "scoring_func": "sigmoid",
  "tie_word_embeddings": false,
  "topk_group": 4,
  "topk_method": "noaux_tc",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.44.2",
  "use_cache": true,
  "v_head_dim": 128,
  "vocab_size": 129280
}
```

## 参数详解

### 1. 基础模型架构与规模

| 参数 | 值 | 说明 |
|------|-----|------|
| `architectures` | `["DeepseekV32ForCausalLM"]` | 指定了加载该模型时使用的类名，这是一个因果语言模型（Causal LM） |
| `model_type` | `"deepseek_v32"` | Hugging Face transformers 库中用来识别模型类型的标识符 |
| `vocab_size` | `129280` | 词表大小，即模型能识别的不同 token 的数量 |
| `hidden_size` | `7168` | 模型的隐藏层维度（embedding dimension），是模型内部表示的核心宽度 |
| `num_hidden_layers` | `61` | Transformer 层的总数量（深度） |
| `max_position_embeddings` | `163840` | 模型支持的最大上下文长度（约为 160k token） |
| `torch_dtype` | `"bfloat16"` | 模型的默认张量数据类型，bfloat16 是大模型训练常用的为了保持数值稳定性兼顾显存的格式 |

### 2. MLA（Multi-Head Latent Attention）

DeepSeek 使用了一种特殊的 MLA 架构来压缩 KV 缓存（KV Cache），这是其核心创新之一。

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_attention_heads` | `128` | 注意力头的数量 |
| `num_key_value_heads` | `128` | 键值（KV）头的数量 |
| `q_lora_rank` | `1536` | **MLA 特有**。对 Query 进行低秩压缩（LoRA）时的 Rank 大小 |
| `kv_lora_rank` | `512` | **MLA 特有**。对 Key-Value 进行低秩压缩时的 Rank 大小，能显著减小 KV Cache |
| `qk_nope_head_dim` | `128` | **MLA 特有**。Query/Key 中不应用旋转位置编码（RoPE）的部分的维度 |
| `qk_rope_head_dim` | `64` | **MLA 特有**。Query/Key 中应用旋转位置编码（RoPE）的部分的维度 |
| `v_head_dim` | `128` | Value 头的维度 |
| `attention_bias` | `false` | 注意力层中的线性变换是否使用偏置项（Bias），通常为了加速设为 false |
| `attention_dropout` | `0.0` | 注意力权重的 Dropout 概率，推理时通常为 0 |

:::info MLA 的核心思想
MLA 将 RoPE 解耦，只对部分维度应用位置编码（`qk_rope_head_dim: 64`），另一部分维度不应用（`qk_nope_head_dim: 128`）。同时通过低秩压缩（LoRA）显著降低 KV Cache 的大小。
:::

### 3. DeepSeekMoE（混合专家模型）

DeepSeekMoE 使用了细粒度专家和共享专家的策略：

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_routed_experts` | `256` | 路由专家（Routed Experts）的总数，模型会从中动态选择一部分 |
| `n_shared_experts` | `1` | 共享专家（Shared Experts）的数量，这些专家对所有 token 都会激活，旨在捕获通用知识 |
| `num_experts_per_tok` | `8` | 路由过程中，每个 token 选择的专家数量 |
| `moe_intermediate_size` | `2048` | 每个 MoE 专家内部前馈网络（FFN）的中间层维度 |
| `moe_layer_freq` | `1` | MoE 层出现的频率，1 表示每一层都是 MoE 层 |
| `intermediate_size` | `18432` | 注意力层之外的密集（Dense）层的中间维度 |
| `first_k_dense_replace` | `3` | 前 3 层被替换为标准 Dense 层，而不使用 MoE，这是为了保持底层表示的稳定性 |
| `n_group` | `8` | 专家分组的数量，用于负载均衡或路由限制 |
| `topk_group` | `4` | 从多少个组中选择专家 |
| `topk_method` | `"noaux_tc"` | Top-K 的选择方法策略 |
| `routed_scaling_factor` | `2.5` | 路由缩放因子 |
| `scoring_func` | `"sigmoid"` | 门控评分函数 |
| `norm_topk_prob` | `true` | 是否归一化 top-k 概率 |

### 4. 位置编码（RoPE + YaRN）

| 参数 | 值 | 说明 |
|------|-----|------|
| `rope_theta` | `10000` | RoPE 的基频 |
| `rope_scaling.type` | `"yarn"` | 使用 YaRN 方法进行长文本扩展 |
| `rope_scaling.factor` | `40` | 扩展倍数 |
| `rope_scaling.original_max_position_embeddings` | `4096` | 原始训练的长度 |
| `rope_scaling.beta_fast` | `32` | YaRN 快速衰减参数 |
| `rope_scaling.beta_slow` | `1` | YaRN 慢速衰减参数 |
| `rope_scaling.mscale` | `1.0` | 缩放因子 |

### 5. FP8 量化配置

模型使用了 FP8（8 位浮点）量化来加速推理并减少显存占用：

| 参数 | 值 | 说明 |
|------|-----|------|
| `quant_method` | `"fp8"` | 量化方法 |
| `fmt` | `"e4m3"` | FP8 格式（4 位指数，3 位尾数），适合权重 |
| `scale_fmt` | `"ue8m0"` | 缩放格式 |
| `activation_scheme` | `"dynamic"` | 激活值动态量化 |
| `weight_block_size` | `[128, 128]` | 权重分块大小 |

### 6. 多 Token 预测（MTP）

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_nextn_predict_layers` | `1` | DeepSeek-V3 的核心训练特性。模型不仅预测下一个 token，还预测后续的 N 个 token |

### 7. 其他常规参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `bos_token_id` | `0` | 开始 token 的 ID |
| `eos_token_id` | `1` | 结束 token 的 ID |
| `hidden_act` | `"silu"` | 激活函数使用 SiLU (Swish) |
| `initializer_range` | `0.02` | 权重初始化的标准差 |
| `rms_norm_eps` | `1e-06` | RMSNorm 层的 epsilon 值，防止除零 |
| `tie_word_embeddings` | `false` | 是否共享输入和输出的 embedding 矩阵 |
| `use_cache` | `true` | 推理时是否缓存 KV 以加速生成 |

## 总参数量计算

DeepSeek-V3 的总参数量约为 **671B（6710 亿）**，而每次推理时的激活参数量约为 **37B（370 亿）**。

### 1. 词嵌入与输出层（Embeddings）

**计算公式：** $\text{Vocab} \times \text{Dim}$

由于 `tie_word_embeddings` 为 `false`，输入和输出矩阵是独立的，需要算两次：

$$129,280 \times 7,168 \times 2 \approx 1.85B$$

### 2. 注意力层（MLA Attention）- 共 61 层

DeepSeek 使用 MLA 架构，参数比标准 MHA 少很多：

- **Query 部分：** 降维压缩 ($d_{model} \to 1536$) + 升维到头 ($1536 \to 128 \times (128+64)$)
- **KV 部分：** 降维压缩 ($d_{model} \to 512$) + 升维到头 ($512 \to 128 \times (128+128)$)
- **输出投影：** $(128 \times 128) \to d_{model}$

单层 Attention 参数约为 **0.19B**，61 层总计：

$$61 \times 0.19 \approx 11.6B$$

### 3. 前 3 层：密集前馈网络（Dense FFN）

这 3 层没有使用专家，是大号的普通层。

**结构：** Gate + Up + Down 三个矩阵（SwiGLU）

**单层计算：** $3 \times d_{model}(7168) \times d_{inter}(18432)$

单层参数约 **0.4B**，3 层总计：

$$3 \times 0.4 \approx 1.2B$$

### 4. 后 58 层：MoE 专家层（MoE FFN）- 参数大头

这部分包含 58 层，每层包含 1 个共享专家 + 256 个路由专家。这是模型总参数膨胀的来源。

- **单个专家参数：** $3 \times 7168 \times 2048 \approx 0.044B$
- **单层所有专家：** $(1 \text{ Shared} + 256 \text{ Routed}) \times 0.044B \approx 11.3B$
- **58 层总计：** $58 \times 11.3B \approx 655.4B$

### 汇总

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embeddings | ~1.9B | 0.3% |
| Attention | ~11.6B | 1.7% |
| Dense FFN (×3) | ~1.2B | 0.2% |
| MoE FFN (×58) | ~655.4B | **97.8%** |
| **总计（Total）** | **~671B** | 100% |

### 激活参数量（Active Params）

虽然总共有 671B 参数，但推理每个 token 时，MoE 层只激活 **8 个路由专家 + 1 个共享专家**：

- **单层 MoE 激活：** $9 \text{ experts} \times 0.044B = 0.396B$（这与 Dense 层的大小惊人一致，设计得很精妙，保证了层间计算负载均衡）
- **总激活：** $1.85B(\text{Emb}) + 61 \times (0.19B \text{ Attn} + 0.396B \text{ FFN}) \approx 37B$

:::tip 关键结论
- **硬盘/显存占用（FP8）：** 取决于 671B 参数（约 600-700GB）
- **计算速度/推理成本：** 等效于一个 37B 的模型
:::

## 架构总结

DeepSeek-V3.2 是一个极其庞大且先进的模型，结合了多项核心技术：

1. **DeepSeekMoE：** 大量细粒度专家（256 个）+ 共享专家，实现参数规模与计算效率的平衡
2. **MLA：** 极致压缩 KV Cache，降低推理显存需求
3. **FP8 量化：** 进一步降低显存占用和提升推理速度
4. **YaRN 长窗口：** 支持 160k token 的超长上下文

这是一个为高性能推理和超长上下文优化的万亿级参数 MoE 模型。

## 参考资料

- [DeepSeek-V3 技术报告](https://github.com/deepseek-ai/DeepSeek-V3)
- [Hugging Face DeepSeek-V3.2 模型页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [MLA: Multi-Head Latent Attention 论文](https://arxiv.org/abs/2405.04434)
