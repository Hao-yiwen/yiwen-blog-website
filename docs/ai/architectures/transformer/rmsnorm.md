---
title: RMSNorm (Root Mean Square Normalization)
sidebar_label: RMSNorm 均方根归一化
date: 2025-12-07
last_update:
  date: 2025-12-07
tags: [transformer, normalization, rmsnorm, layernorm, llama, llm]
---

# 技术文档：RMSNorm (Root Mean Square Normalization)

**RMSNorm (Root Mean Square Normalization)** 可以被视为 **LayerNorm (Layer Normalization)** 的"简化版"或"加速版"。

目前像 LLaMA、PaLM、Gopher 等主流大模型纷纷用 RMSNorm 取代 LayerNorm，主要是为了在**保持模型性能（效果）几乎不变的前提下，显著提升计算速度和训练稳定性**。

---

## 1. 核心区别：算不算"均值"

两者最大的区别在于：**是否对数据进行了"去中心化"（Mean Centering）**。

### LayerNorm (LN)

传统的 LayerNorm 做两件事：

1.  **去中心化（Re-centering）：** 算出输入的均值，然后减去它（让数据均值为 0）。
2.  **缩放（Re-scaling）：** 算出方差，除以标准差（让数据方差为 1）。

**公式：**

$$\bar{x} = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$$

*(其中 $\mu$ 是均值，$\sigma$ 是标准差)*

### RMSNorm

RMSNorm 认为"去中心化"这一步是多余的，它只做一件事：

1.  **缩放（Re-scaling）：** 直接根据输入的均方根（RMS）进行缩放。

**公式：**

$$\bar{x} = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma$$

*(其中 $\text{RMS}(x) = \sqrt{\frac{1}{n} \sum x_i^2}$，不需要计算均值 $\mu$)*

---

## 2. 为什么现在 RMSNorm 替代了 LayerNorm？

在大模型时代，这个替代主要由以下三个原因驱动：

### A. 计算效率更高 (Speed & Efficiency)

这是最主要的原因。

  * **少算一步：** LayerNorm 需要先算均值，再减均值，再算方差。RMSNorm 直接算均方根。虽然单次计算看起来省得不多，但在一个几百亿参数、几十层深的大模型中，这个操作要重复数十亿次，累计节省的计算时间（Wall-clock time）非常可观。
  * **节省显存带宽：** 现代 GPU 训练通常受限于显存带宽（Memory Bandwidth）而不是计算能力。RMSNorm 的计算过程更简单，减少了数据的搬运和读写。
  * **实测数据：** 研究表明，在某些 Transformer 模型中，使用 RMSNorm 可以带来 **10% ~ 40%** 的推理加速。

### B. 效果并没有变差 (Performance)

这一条是替代的前提。

  * Geoffrey Hinton 等人的研究以及后续的大量实验发现，LayerNorm 起作用的关键在于**缩放（Scaling）**的不变性，而不是**平移（Shifting/Centering）**的不变性。
  * 也就是说，**"减去均值"这一步对于深层网络的训练稳定性贡献很小**。既然去掉了也不影响精度，为了速度自然就去掉了。

### C. 简化参数 (Simplification)

  * 传统的 LayerNorm 通常有两个可学习参数：缩放因子 $\gamma$ (gain) 和 平移因子 $\beta$ (bias)。
  * 主流的 RMSNorm 实现（如 LLaMA 所用的）通常**去掉了 Bias ($\beta$) 项**，只保留缩放因子 $\gamma$。这进一步减少了参数量和内存占用，且在某些情况下能提升模型的数值稳定性。

---

## 3. 代码对比 (PyTorch)

看代码会更直观，你会发现 RMSNorm 少了一大块逻辑：

```python
import torch

# === LayerNorm ===
# 1. 计算均值 (Mean)
mean = x.mean(dim=-1, keepdim=True)
# 2. 计算方差 (Variance)
var = x.var(dim=-1, keepdim=True, unbiased=False)
# 3. 归一化 (减均值，除标准差)
x_norm = (x - mean) / torch.sqrt(var + epsilon)
# 4. 仿射变换 (乘 gamma 加 beta)
output = x_norm * gamma + beta


# === RMSNorm ===
# 1. 不算均值，直接算均方根 (RMS)
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
# 2. 归一化 (只除以 RMS)
x_norm = x / rms
# 3. 仿射变换 (通常只有 gamma，没有 beta)
output = x_norm * gamma
```

---

## 4. 完整的 RMSNorm 实现

以下是一个可用于生产环境的 RMSNorm 模块实现：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    论文: https://arxiv.org/abs/1910.07467

    Args:
        dim (int): 输入的最后一个维度大小
        eps (float): 防止除零的小常数，默认 1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算 RMS 归一化"""
        # x.pow(2).mean(-1, keepdim=True): 计算每个样本的均方值
        # rsqrt: 计算平方根的倒数 (1/sqrt(x))，比 1/torch.sqrt(x) 更快
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先转为 float32 计算以保证数值精度，再转回原始 dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# --- 使用示例 ---
if __name__ == '__main__':
    batch_size, seq_len, d_model = 2, 128, 4096

    # 初始化 RMSNorm
    rms_norm = RMSNorm(dim=d_model)

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = rms_norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
    print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
```

---

## 5. 总结对比

| 特性 | LayerNorm | RMSNorm |
| :--- | :--- | :--- |
| **操作** | 减均值 + 除标准差 | **仅**除以均方根 |
| **计算量** | 较高 (计算 $\mu$ 和 $\sigma$) | **较低** (省去了 $\mu$) |
| **参数** | 缩放 $\gamma$ + 偏置 $\beta$ | 通常只有缩放 $\gamma$ |
| **主要优势** | 理论完备，早期标准 | **速度快，显存友好** |
| **代表模型** | BERT, GPT-2, RoBERTa | **LLaMA, T5, Gopher** |

**一句话总结：** RMSNorm 是 LayerNorm 的"省流版"，它去掉了对大模型训练帮助不大的"减均值"操作，换来了更快的速度和更简单的实现，是目前大模型追求极致效率的必然选择。
