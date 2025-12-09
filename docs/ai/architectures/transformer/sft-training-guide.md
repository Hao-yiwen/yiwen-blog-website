---
title: SFT 有监督微调训练指南
sidebar_position: 24
tags: [SFT, 微调, LLM训练, Instruction Tuning, RLHF, PPO]
---

# SFT (Supervised Fine-Tuning)

> **摘要**：SFT 全称 Supervised Fine-Tuning（有监督微调），是目前大语言模型（LLM）从"续写小说"进化为"听懂指令的助手"的关键步骤。简单来说，SFT 就是给一个已经读过万卷书（预训练）但不懂规矩的"天才学生"，发一本"标准问答习题集"，手把手教它怎么正确回答问题。

-----

## 1. SFT 在 LLM 训练流程中的位置

SFT 是连接"预训练"和"人类对齐"的桥梁。一个完整的 ChatGPT 类模型的训练通常包含三个阶段，SFT 处于第二阶段：

### 1.1 Pre-training (预训练)

- **目标**: 让模型学会语言规律和世界知识（海量文本）。
- **状态**: 此时的模型像个"成语接龙"高手，你说"你好"，它可能会接"你好吗？你好在哪里..."，而不是回答你。

### 1.2 SFT (有监督微调)

- **目标**: **学会听指令**。让模型学会"用户提问 -> 模型回答"的对话模式。
- **状态**: 经过 SFT，模型变成了"助手"，你说"你好"，它会回答"你好！有什么我可以帮你的吗？"。

### 1.3 RLHF-PPO (人类反馈强化学习)

- **目标**: 对齐人类价值观（更安全、更有用、更诚实）。
- **状态**: 进一步优化回答的质量，减少有害信息。

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 训练三阶段流程图                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Pre-train   │───▶│     SFT      │───▶│  RLHF-PPO    │      │
│  │   (预训练)    │    │  (有监督微调)  │    │ (强化学习对齐) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│    学会语言规律          学会听指令           对齐人类价值观       │
│    "成语接龙"           "问答助手"           "安全有用"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

-----

## 2. SFT 的核心原理

SFT 的本质依然是**"下一个 Token 预测（Next Token Prediction）"**，但在训练数据和损失函数（Loss）计算上与预训练有重要区别。

### 2.1 数据格式 (Instruction-Response Pairs)

SFT 需要高质量的**"指令-回复"对**数据。最经典的格式是 JSON 形式：

```json
{
  "instruction": "请把下面的句子翻译成英文。",
  "input": "今天天气真好。",
  "output": "The weather is really good today."
}
```

### 2.2 训练过程与 Loss Masking (关键技术点)

在预训练时，模型会对所有文本计算 Loss（即学习每一个字）。但在 SFT 中，我们通常**只关心模型回答得对不对**，而不关心它能不能复述问题。

- **Prompt (用户指令)**: `请把下面的句子翻译成英文...` -> **不计算 Loss** (Masked)
- **Response (模型回答)**: `The weather is really good today.` -> **计算 Loss**

这意味着模型在训练时，虽然"看"到了问题，但只有在生成回答的那些 token 时，才会被更新参数。这强迫模型集中精力学习"如何生成正确的回答"。

```
┌────────────────────────────────────────────────────────────┐
│                    Loss Masking 示意图                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  输入序列: [User: 1+1=?] [Assistant:] [2]                   │
│                                                            │
│  Labels:   [-100][-100][-100][-100][-100][-100][2]         │
│            ▲                               ▲               │
│            │                               │               │
│         Masked                        计算 Loss            │
│        (忽略不学)                      (重点学习)            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

-----

## 3. SFT 的数据来源

SFT 的效果高度依赖数据质量（**Quality > Quantity**）。通常几千到几万条高质量数据，比几百万条低质量数据更有效。

| 数据来源 | 说明 | 示例 |
|:---------|:-----|:-----|
| **人工标注** | 由专业人员编写高质量的问答 | OpenAI 早期雇佣博士写数据 |
| **模型蒸馏 (Self-Instruct)** | 让更强的模型（如 GPT-4）生成指令和回复 | Alpaca, Vicuna |
| **特定领域数据** | 医疗报告、法律文书、代码库等清洗后的专业数据 | 医疗问诊、代码补全 |

-----

## 4. SFT vs. RLHF-PPO 的区别

| 特性 | SFT (有监督微调) | RLHF-PPO (人类反馈强化学习) |
|:-----|:-----------------|:---------------------------|
| **核心逻辑** | **模仿学习** (Imitation Learning) | **奖惩学习** (Preference Learning) |
| **比喻** | 老师写好标准答案，学生照抄背诵 | 学生写出几个答案，老师打分，学生自己琢磨怎么拿高分 |
| **目的** | 学会格式、语气、基本指令遵循 | 学会判别好坏、减少幻觉、对齐价值观 |
| **数据需求** | 需要**标准答案** | 需要**排序数据** (A比B好) |
| **难度** | 相对简单，稳定 | 较难，训练不稳定，容易跑偏 |

-----

## 5. RLHF-PPO 详解

### 5.1 什么是 RLHF？

RLHF（Reinforcement Learning from Human Feedback）即**人类反馈强化学习**，是让 LLM 对齐人类价值观的关键技术。PPO（Proximal Policy Optimization）是其中最常用的强化学习算法。

### 5.2 RLHF-PPO 的训练流程

RLHF-PPO 需要同时维护 **4 个模型**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF-PPO 四模型架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │    Actor     │    │   Critic     │                          │
│  │  (策略模型)   │    │  (价值模型)   │                          │
│  │   可训练      │    │   可训练      │                          │
│  └──────────────┘    └──────────────┘                          │
│         │                   │                                   │
│         ▼                   ▼                                   │
│     生成回答              估计奖励                                │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │   Reward     │    │  Reference   │                          │
│  │  (奖励模型)   │    │  (参考模型)   │                          │
│  │   冻结       │    │   冻结        │                          │
│  └──────────────┘    └──────────────┘                          │
│         │                   │                                   │
│         ▼                   ▼                                   │
│     打分评价              KL 惩罚                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 训练步骤：

1. **Rollout (采样)**：Actor 模型根据 Prompt 生成回答
2. **Reward (打分)**：Reward Model 对生成的回答打分
3. **KL Penalty (约束)**：计算 Actor 与 Reference 的 KL 散度，防止模型跑偏
4. **Update (更新)**：用 PPO 算法更新 Actor 和 Critic

### 5.3 PPO 的核心公式

PPO 的目标函数：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是裁剪参数（通常 0.1~0.2）

### 5.4 RLHF-PPO 的痛点

| 痛点 | 说明 |
|:-----|:-----|
| **太复杂** | 需要同时维护 4 个模型，工程难度大 |
| **不稳定** | 超参数极其敏感，训练经常不收敛 |
| **极慢** | 需要实时生成（Rollout）文本，不仅慢，还极其吃显存 |
| **显存爆炸** | 4 个模型同时加载，7B 模型需要 80GB+ 显存 |

### 5.5 PPO 的简化代码示例

```python
import torch
import torch.nn.functional as F

def ppo_loss(
    old_log_probs,      # 旧策略的 log 概率
    new_log_probs,      # 新策略的 log 概率
    advantages,         # 优势函数
    clip_epsilon=0.2    # 裁剪参数
):
    """
    PPO 的核心 Loss 计算
    """
    # 计算概率比 r(θ) = π_new / π_old
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 未裁剪的目标
    surr1 = ratio * advantages

    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    # PPO Loss：取两者最小值（保守策略）
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_kl_penalty(log_probs_policy, log_probs_ref, kl_coef=0.1):
    """
    计算 KL 散度惩罚，防止模型偏离参考模型太远
    """
    kl_div = torch.exp(log_probs_ref) * (log_probs_ref - log_probs_policy)
    return kl_coef * kl_div.sum(dim=-1).mean()
```

-----

## 6. SFT 的优缺点

### 优点

- **见效快**：只要有几百条高质量数据，就能迅速改变模型的说话风格（例如把一个通用模型微调成"红楼梦林黛玉风格"）
- **效率高**：相比预训练，SFT 消耗的算力极小（通常只需几个小时到几天）
- **稳定**：训练过程稳定，不像 PPO 那样容易跑偏

### 缺点

- **幻觉问题**：如果训练数据里包含错误的事实，模型会照单全收并"一本正经地胡说八道"
- **能力天花板**：SFT 只是模仿，很难超越提供数据的标注者的水平
- **遗忘风险**：过度微调可能导致模型忘记预训练中学到的通用知识（Catastrophic Forgetting）

-----

## 7. SFT 核心代码实现 (PyTorch)

以下是一个完整的 SFT 训练代码示例，展示了 Loss Masking 的核心逻辑：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

# ==========================================
# 1. 准备一份极简的 SFT 数据 (Instruction格式)
# ==========================================
sft_data = [
    {"instruction": "这只猫叫什么名字？", "output": "这只猫叫咪咪。"},
    {"instruction": "1加1等于几？", "output": "1加1等于2。"},
    {"instruction": "天空是什么颜色的？", "output": "通常是蓝色的。"},
]

# ==========================================
# 2. 定义 Dataset 和 核心 Masking 逻辑
# ==========================================
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 设置填充token，GPT2默认没有pad_token，这里用eos_token代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]

        # A. 构建输入文本： Prompt + Answer + EOS
        # 不同的模型有不同的Prompt模版，这里演示最简单的拼接
        prompt_text = f"User: {instruction}\nAssistant: "
        full_text = prompt_text + output + self.tokenizer.eos_token

        # B. 分词 (Tokenization)
        # 1. 对完整的序列进行编码
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = full_encoding["input_ids"][0]
        attention_mask = full_encoding["attention_mask"][0]

        # 2. 关键步骤：找到 Prompt 在序列中的长度，以便 Mask 掉它
        # 这里我们单独对 prompt 进行编码，看看它有多长
        prompt_encoding = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        # C. 构建 Labels (核心中的核心)
        # 初始 Labels 是 Input_ids 的副本
        labels = input_ids.clone()

        # 1. 将 Prompt 部分的 Label 设为 -100 (PyTorch Loss 会忽略这些)
        # 注意：要防止 prompt_len 超过截断后的总长度
        safe_prompt_len = min(prompt_len, len(labels))
        labels[:safe_prompt_len] = -100

        # 2. 将 Padding 部分 (原本是 pad_token_id) 也设为 -100
        # attention_mask 为 0 的地方就是 padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ==========================================
# 3. 初始化模型和加载器
# ==========================================
# 使用 gpt2 举例，实际中可能是 Llama3 或 Qwen
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

dataset = SFTDataset(sft_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ==========================================
# 4. 手写训练循环 (Training Loop)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

print(f"开始训练，使用设备: {device}")
print("-" * 30)

epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        # 1. 数据搬运到 GPU/CPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. 前向传播 (Forward Pass)
        # 传入 labels 后，HuggingFace 模型内部会自动计算 CrossEntropyLoss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # 4. 反向传播 (Backward)
        loss.backward()

        # 5. 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# ==========================================
# 5. 简单的推理测试
# ==========================================
print("-" * 30)
print("测试训练后的模型：")
model.eval()
test_question = "User: 1加1等于几？\nAssistant: "
inputs = tokenizer(test_question, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

-----

## 8. 实战建议

### 8.1 数据准备

- **质量优先**：1000 条高质量数据 > 100000 条低质量数据
- **多样性**：覆盖不同类型的指令（问答、翻译、总结、代码等）
- **格式统一**：使用一致的 Prompt 模板

### 8.2 训练技巧

| 技巧 | 说明 |
|:-----|:-----|
| **学习率** | 通常 1e-5 ~ 5e-5，比预训练小 |
| **Batch Size** | 尽量大，提高训练稳定性 |
| **LoRA** | 对于大模型（7B+），使用 LoRA 可大幅降低显存 |
| **Early Stopping** | 监控验证集 Loss，防止过拟合 |

### 8.3 常见问题

| 问题 | 解决方案 |
|:-----|:---------|
| 模型胡说八道 | 检查数据质量，可能有错误信息 |
| Loss 不下降 | 学习率可能太小，或数据格式有问题 |
| 遗忘旧知识 | 降低学习率，或混入部分预训练数据 |
| 回答风格不对 | 检查 Prompt 模板是否正确 |

-----

## 9. 总结

SFT 是将"读死书"的模型转化为"好用工具"的点金石。如果你想打造一个垂直领域的专属模型（比如公司内部的客服机器人、法律助手），**SFT 是你最需要关注和操作的环节。**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SFT + RLHF-PPO 黄金组合                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │     SFT      │  ────▶  │  RLHF-PPO    │                     │
│  │   学会说话    │         │   说得更好    │                     │
│  └──────────────┘         └──────────────┘                     │
│         │                        │                              │
│         ▼                        ▼                              │
│   模仿标准答案              对齐人类偏好                          │
│   格式/语气/知识             安全/有用/诚实                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

在 2024 年及以后，**SFT + DPO/PPO** 几乎是定制大模型的标准流程。SFT 奠定基础，对齐方法（DPO 或 PPO）锦上添花。
