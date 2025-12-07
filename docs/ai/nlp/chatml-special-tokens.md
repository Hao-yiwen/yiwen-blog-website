---
title: ChatML 格式与常用特殊标记
sidebar_label: ChatML 特殊标记
date: 2025-12-07
last_update:
    date: 2025-12-07
tags: [ChatML, Special Tokens, LLM, SFT]
---

# ChatML 格式与常用特殊标记

## 什么是 ChatML

**ChatML (Chat Markup Language)** 是一种将多轮对话序列化为模型输入的标准格式。最早由 OpenAI 提出，现在被 Qwen、Yi、DeepSeek 等众多模型广泛采用。

核心思想：**把对话变成一种类似 HTML/XML 的文档结构**，使用特殊标记来区分不同角色的发言。

### 基本结构

```text
<|im_start|>system
你是一个专业的AI助手。<|im_end|>
<|im_start|>user
你好！<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>
```

### 结构拆解

| 组成部分 | 文本内容 | 说明 |
|:---|:---|:---|
| **Header** | `<\|im_start\|>system\n` | 声明接下来是**系统指令** |
| **Content** | `你是一个专业的AI助手。` | 系统提示词 (System Prompt) |
| **Footer** | `<\|im_end\|>\n` | 系统指令结束 |
| **Header** | `<\|im_start\|>user\n` | 声明接下来是**用户发言** |
| **Content** | `你好！` | 用户的问题 |
| **Footer** | `<\|im_end\|>\n` | 用户发言结束 |
| **Header** | `<\|im_start\|>assistant\n` | 声明接下来是**AI发言** |
| **Content** | `你好！有什么...` | AI 的回答（SFT 训练目标） |
| **Footer** | `<\|im_end\|>\n` | AI 回答结束 (EOS) |

---

## 常用特殊标记 (Special Tokens)

### 1. 通用特殊标记

这些标记在几乎所有 LLM 中都存在：

| 标记 | 名称 | 作用 | 示例 |
|:---|:---|:---|:---|
| `<bos>` / `<s>` | Begin of Sequence | 序列开始标记 | 表示一段文本的起始 |
| `<eos>` / `</s>` | End of Sequence | 序列结束标记 | 告诉模型停止生成 |
| `<pad>` | Padding | 填充标记 | 批处理时对齐不同长度的序列 |
| `<unk>` | Unknown | 未知标记 | 表示词表中不存在的 token |

### 2. ChatML 特殊标记

ChatML 格式的核心标记：

| 标记 | 含义 | 用途 |
|:---|:---|:---|
| `<\|im_start\|>` | **I**nstruction **M**essage Start | 消息开始，后接角色名 |
| `<\|im_end\|>` | **I**nstruction **M**essage End | 消息结束，充当 EOS |
| `system` | 系统角色 | 设定 AI 的行为和背景 |
| `user` | 用户角色 | 用户的输入 |
| `assistant` | 助手角色 | AI 的回复 |

### 3. 不同模型的特殊标记对比

#### Qwen / Qwen2 系列

```text
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
今天天气怎么样？<|im_end|>
<|im_start|>assistant
```

| 标记 | Token ID（示例） |
|:---|:---|
| `<\|im_start\|>` | 151644 |
| `<\|im_end\|>` | 151645 |
| `<\|endoftext\|>` | 151643 |

#### LLaMA 2 / LLaMA 3 系列

**LLaMA 2 格式：**
```text
<s>[INST] <<SYS>>
你是一个有帮助的助手。
<</SYS>>

今天天气怎么样？ [/INST] 今天天气晴朗。</s>
```

**LLaMA 3 格式：**
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个有帮助的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>

今天天气怎么样？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

| 标记 | 作用 |
|:---|:---|
| `<\|begin_of_text\|>` | 文本开始 (BOS) |
| `<\|end_of_text\|>` | 文本结束 (EOS) |
| `<\|start_header_id\|>` | 角色标识开始 |
| `<\|end_header_id\|>` | 角色标识结束 |
| `<\|eot_id\|>` | End of Turn，轮次结束 |

#### Yi 系列

Yi 模型采用标准 ChatML 格式：

```text
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

#### Mistral / Mixtral 系列

```text
<s>[INST] 你是一个有帮助的助手。

今天天气怎么样？ [/INST] 今天天气晴朗。</s>
```

| 标记 | 作用 |
|:---|:---|
| `<s>` | 序列开始 (BOS) |
| `</s>` | 序列结束 (EOS) |
| `[INST]` | 指令开始 |
| `[/INST]` | 指令结束 |

---

## 特殊标记汇总表

| 模型系列 | BOS | EOS | 消息开始 | 消息结束 | 轮次结束 |
|:---|:---|:---|:---|:---|:---|
| **Qwen/Yi** | - | `<\|endoftext\|>` | `<\|im_start\|>` | `<\|im_end\|>` | `<\|im_end\|>` |
| **LLaMA 3** | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | `<\|start_header_id\|>` | `<\|end_header_id\|>` | `<\|eot_id\|>` |
| **LLaMA 2** | `<s>` | `</s>` | `[INST]` | `[/INST]` | `</s>` |
| **Mistral** | `<s>` | `</s>` | `[INST]` | `[/INST]` | `</s>` |

---

## SFT 训练中的 Loss Mask

结合 **Loss Masking**，训练时只对 assistant 的回复计算损失：

```text
[Token]           [Mask]
<|im_start|>        0
system              0
\n                  0
你是AI助手...        0
<|im_end|>          0
\n                  0
<|im_start|>        0
user                0
\n                  0
SFT是什么...         0
<|im_end|>          0     <-- Prompt 结束
\n                  0
<|im_start|>        0
assistant           0     <-- 标签也是 0
\n                  0
SFT                 1     <-- 开始算分！
是                  1
Supervised          1
...                 1
微调                1
。                  1
<|im_end|>          1     <-- 必须学会预测结束符
```

**关键点：**
- `Mask = 0`：不计算损失（Prompt 部分）
- `Mask = 1`：计算损失（模型需要学习生成的部分）
- `<|im_end|>` 也需要学习，否则模型不知道何时停止

---

## 代码示例

### 使用 HuggingFace Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

messages = [
    {"role": "system", "content": "你是一个专业的AI助手。"},
    {"role": "user", "content": "SFT是什么意思？"},
    {"role": "assistant", "content": "SFT是有监督微调。"}
]

# 自动转换成 ChatML 格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print(text)
```

### 输出示例

```text
<|im_start|>system
你是一个专业的AI助手。<|im_end|>
<|im_start|>user
SFT是什么意思？<|im_end|>
<|im_start|>assistant
SFT是有监督微调。<|im_end|>
```

### 生成时添加 Prompt

```python
# 推理时：只有 system + user，让模型生成 assistant
messages = [
    {"role": "system", "content": "你是一个专业的AI助手。"},
    {"role": "user", "content": "什么是强化学习？"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 自动添加 <|im_start|>assistant\n
)
```

---

## 为什么需要特殊标记？

1. **区分角色**：让模型知道谁在说话
2. **控制生成**：EOS 标记告诉模型何时停止
3. **多轮对话**：清晰分隔每一轮对话
4. **训练效率**：精确控制哪些 token 参与损失计算
5. **标准化**：不同工具和框架可以统一处理

---

## 参考资料

- [OpenAI ChatML 规范](https://github.com/openai/openai-python/blob/main/chatml.md)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [LLaMA 3 Model Card](https://github.com/meta-llama/llama3)
