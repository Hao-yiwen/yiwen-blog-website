---
title: PyTorch 文件格式详解
sidebar_label: 文件格式 (.pt/.pth/.safetensors)
date: 2025-12-07
tags: [pytorch, safetensors, huggingface, 模型存储]
---

# PyTorch 文件格式详解：.pt、.pth 与 .safetensors

`.pt`、`.pth` 和 `.safetensors` 是深度学习中最常见的模型文件格式。它们代表的技术路线和安全级别完全不同。

**一句话总结：** `.pt` 和 `.pth` 是亲兄弟（都是 PyTorch 的原生格式），而 `.safetensors` 是为了解决它俩的缺陷而生的"新一代标准"。

## 1. `.pt` 和 `.pth`：本质相同

这两者**没有任何技术上的区别**，它们都是 PyTorch 使用 Python 的 `pickle` 模块序列化后的文件。

### 全称

- `.pt`: PyTorch
- `.pth`: PyTorch

### 本质

它们都是用 `torch.save()` 生成的：

```python
import torch

# 保存模型权重
torch.save(model.state_dict(), "model.pt")
# 或者
torch.save(model.state_dict(), "model.pth")

# 加载模型权重
state_dict = torch.load("model.pt")
model.load_state_dict(state_dict)
```

### 区别

纯粹是**命名习惯**不同：

- 有些人喜欢用 `.pt`（官方文档现在更倾向于这个，因为更短）
- 有些人喜欢用 `.pth`（早期的习惯）
- Hugging Face 早期的 PyTorch 模型通常命名为 `pytorch_model.bin`，本质上也是 `.pt` 格式

:::warning 小坑提示
在 Windows 系统上，`.pth` 有时会被误认为是 Python 的路径配置文件（Path configuration file），导致一些小麻烦，所以现在的习惯趋向于用 `.pt`。
:::

## 2. Pickle 格式的问题

既然 `.pt` 挺好用，Hugging Face 为什么要费劲搞一个新格式 `.safetensors`？

因为 `.pt` (Pickle) 有两个**致命死穴**：

### 死穴一：不安全（核心原因）

`.pt` 文件基于 Python 的 `pickle` 模块。`pickle` 是一个非常强大的序列化工具，它不仅能存数据，还能存**代码逻辑**。

**黑客攻击场景：**

1. 黑客写一段恶意代码（比如删除你硬盘文件，或者窃取你的 SSH 密钥）
2. 黑客把这段代码包装进一个大模型的 `.pt` 权重文件里
3. 你开心地下载了 `Llama-3-Hacked.pt`
4. 当你运行 `torch.load("Llama-3-Hacked.pt")` 的那一瞬间，**恶意代码自动执行**，你的电脑就中招了

:::danger 安全警告
下载别人的 `.pt` 文件，就像运行别人发给你的 `.exe` 程序一样危险。
:::

### 死穴二：加载慢（显存杀手）

`pickle` 加载数据的方式非常低效，特别是对于几十 GB 的大模型：

1. **CPU 忙死：** 需要 CPU 先把数据"解包" (Unpickle)
2. **内存爆炸：** 数据需要先加载到 CPU 内存（RAM），然后再复制到 GPU 显存（VRAM）。这导致加载 10GB 的模型瞬间可能需要 20GB 的内存峰值

## 3. `.safetensors`：新时代的标准

`.safetensors` 是 Hugging Face 专门为了解决上述问题开发的**纯二进制格式**。

### 对比总结

| 特性 | `.pt` / `.pth` (Pickle) | `.safetensors` |
| :--- | :--- | :--- |
| **安全性** | ❌ 危险：可以包含恶意代码 | ✅ 安全：纯数据文件，绝不执行代码 |
| **加载速度** | 🐢 慢：需要 CPU 解包和拷贝 | ⚡ 极快：利用内存映射 (mmap) |
| **内存占用** | 🟥 高：需要双倍内存 (RAM + VRAM) | 🟩 极低：零拷贝 (Zero-copy) |
| **通用性** | 仅限 Python / PyTorch | 跨语言 (Rust, Python, JS, C++) |
| **懒加载** | ❌ 很难只加载模型的一部分 | ✅ 轻松只加载某几层的权重 (Sharding) |

### 为什么 `.safetensors` 这么快？

它使用了 **Memory Mapping (mmap)** 技术。

简单说，它不需要把文件"读"进内存。它只是告诉操作系统："硬盘上这个文件的这一段数据，直接对应我显存里的这一块地址。"

数据是**直接**从硬盘流向 GPU 的，跳过了 CPU 解包和内存复制的繁琐过程。

```
传统 Pickle 加载流程：
硬盘 → CPU 解包 → RAM → 复制 → VRAM
        ↓
      很慢，内存翻倍

Safetensors 加载流程：
硬盘 → mmap → VRAM
        ↓
      快速，零拷贝
```

## 4. 使用建议

### 你的模型该存成什么？

**场景一：自己训练、自己用（内部环境）**

用 `.pt` 或 `.pth` 完全没问题，方便快捷：

```python
# 保存
torch.save(model.state_dict(), "model.pt")

# 加载
model.load_state_dict(torch.load("model.pt"))
```

**场景二：发布到 Hugging Face 给别人用**

强烈建议保存为 `.safetensors`：

```python
from safetensors.torch import save_file, load_file

# 保存
save_file(model.state_dict(), "model.safetensors")

# 加载
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
```

### Hugging Face Transformers 集成

```python
from transformers import AutoModelForCausalLM

# 保存时使用 safe_serialization=True
model.save_pretrained("./output_dir", safe_serialization=True)
# 会生成 model.safetensors

# 加载时自动识别格式，不用改代码
model = AutoModelForCausalLM.from_pretrained("./output_dir")
```

### 格式转换

从 `.pt` 转换为 `.safetensors`：

```python
import torch
from safetensors.torch import save_file

# 加载 pickle 格式
state_dict = torch.load("model.pt", map_location="cpu")

# 保存为 safetensors 格式
save_file(state_dict, "model.safetensors")
```

## 5. `.safetensors` 的通用性：不只是 LLM

**绝对不只能给 LLM 用！它可以用于任何深度学习模型，包括 CNN、RNN、GAN，甚至是非 AI 的纯数学计算数据。**

`.safetensors` 的本质不是"LLM 存储器"，而是一个**"通用的、安全的、高效的张量（Tensor）存储格式"**。

只要你的模型参数是由**数字矩阵（Tensors）**组成的，它就能存。

### 为什么它能通用？（底层逻辑）

计算机根本不知道什么是"语言模型"什么是"CNN"。

- **LLM (Llama 3):** 权重是一堆形状为 `[4096, 4096]` 的矩阵。
- **CNN (ResNet):** 权重是一堆形状为 `[64, 3, 7, 7]` 的卷积核矩阵。

对于 `.safetensors` 来说，它们都是**由名字（Key）和数组（Value）组成的字典**。它只负责把这些数组安全、快速地存到硬盘上，至于这些数组用来做卷积还是做 Attention，它不在乎。

### 事实证明：AI 绘画领域（CNN/UNet）用得比 LLM 还早

其实，**Stable Diffusion (AI 绘画)** 社区才是推动 `.safetensors` 普及的急先锋。

- **背景：** 在 2022-2023 年，大家疯狂在网上（如 Civitai）下载别人微调好的 `.ckpt` (Pickle 格式) 模型文件。
- **危机：** 有黑客把勒索病毒藏在 `.ckpt` 文件里，导致很多人下载模型后电脑中招。
- **变革：** Civitai 和 Hugging Face 强制推广 `.safetensors`。
- **现状：** 现在你在网上下载的所有 Stable Diffusion 模型（本质是 UNet，一种类似于 CNN 的结构），几乎全是 `.safetensors` 格式。

**结论：AI 绘画（CV领域）早就全面拥抱它了。**

### 代码演示：怎么给 CNN 用？

假设你有一个标准的 ResNet (CNN) 模型，存取方式和 LLM 一模一样。

#### 保存一个 CNN 模型

```python
import torch
import torchvision.models as models
from safetensors.torch import save_file

# 1. 创建一个 CNN 模型 (ResNet18)
cnn_model = models.resnet18(pretrained=True)

# 2. 获取它的 state_dict (就是一堆张量)
state_dict = cnn_model.state_dict()

# 3. 直接保存为 safetensors
# 只要安装了 safetensors 库：pip install safetensors
save_file(state_dict, "resnet18.safetensors")

print("CNN 模型保存成功！")
```

#### 加载一个 CNN 模型

```python
from safetensors.torch import load_file

# 1. 从硬盘读取张量 (速度极快)
loaded_state_dict = load_file("resnet18.safetensors")

# 2. 塞回模型里
cnn_model = models.resnet18()
cnn_model.load_state_dict(loaded_state_dict)

print("CNN 模型加载成功！")
```

### 为什么在 CNN 教程里见得少？

你感觉它好像是 LLM 专属，主要是因为**需求迫切程度**不同：

1. **文件大小不同：**

   - **LLM:** 动不动 20GB、100GB。加载慢是痛点，`.safetensors` 的**零拷贝（Zero-copy）**技术能让加载速度快 10 倍，这是刚需。
   - **CNN:** 经典的 ResNet50 只有 100MB。用 `.pth` 加载也就 0.1 秒，用 `.safetensors` 加载 0.01 秒，用户**体感差异不大**。

2. **生态惯性：**

   - PyTorch 的官方 `torchvision` 库默认还是给 `.pth`。
   - 但是 Hugging Face 的 `timm` (PyTorch Image Models) 库（CV 界最强的库之一）已经开始全面支持 `.safetensors` 了。

### 最佳实践

- **能不能用？** 能，随便用。
- **好不好用？** 好用。更安全，且跨语言（你在 Python 存的 CNN，可以用 Rust 或 C++ 直接读取部署，不需要 Python 环境）。
- **建议：** 如果你在做公司项目，或者要公开发布模型，**强烈建议无论是什么模型（CNN, RNN, Transformer），都统一使用 `.safetensors`**。这是专业度的体现。

## 6. 总结

| 格式 | 推荐场景 | 安全性 | 性能 |
| :--- | :--- | :--- | :--- |
| `.pt` / `.pth` | 个人项目、内部使用 | ⚠️ 需谨慎 | 一般 |
| `.safetensors` | 开源发布、生产环境 | ✅ 安全 | 优秀 |

**一句话：`.pt` 是旧时代的习惯，`.safetensors` 是大模型时代的工业标准。**

## 参考资料

- [Safetensors GitHub](https://github.com/huggingface/safetensors)
- [PyTorch 官方文档 - Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Hugging Face - Safetensors 介绍](https://huggingface.co/docs/safetensors)
