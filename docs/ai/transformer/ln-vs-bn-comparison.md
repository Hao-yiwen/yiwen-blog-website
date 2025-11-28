---
title: 层归一化 (LN) vs 批归一化 (BN)
sidebar_label: LN vs BN 对比
date: 2025-11-28
last_update:
  date: 2025-11-28
tags: [transformer, normalization, layer-norm, batch-norm]
---

# 层归一化 (LN) vs 批归一化 (BN)

**层归一化 (Layer Normalization, LN)** 和 **批归一化 (Batch Normalization, BN)** 是深度学习中两种最常用的归一化技术。它们的核心目的都是为了解决 **"内部协变量偏移" (Internal Covariate Shift)** 问题，从而加速模型收敛并提高训练稳定性。

简单来说，它们的区别在于 **"归一化的维度"** 不同。

## 1. 核心概念对比

为了直观理解，假设我们的输入数据是一个形状为 $[N, C, H, W]$ 的张量（常见于图像处理），其中：
* $N$: 样本数量 (Batch Size)
* $C$: 通道数 (Channel/Feature)
* $H, W$: 特征图的高和宽

### 批归一化 (Batch Normalization - BN)
* **方向：** "纵向"切。
* **操作：** 对 **同一个通道 (Channel)**，利用 **整个 Batch** 的数据计算均值和方差。
* **直觉：** 假设你在分析一个班级的考试成绩。BN 就像是**把全班同学的"数学成绩"拿出来进行标准化**，然后再把"英语成绩"拿出来标准化。它看重的是不同样本在同一特征上的分布。
* **适用场景：** 计算机视觉 (CNN)。

### 层归一化 (Layer Normalization - LN)
* **方向：** "横向"切。
* **操作：** 对 **同一个样本 (Sample)**，利用其 **所有通道 (Channels)** 的数据计算均值和方差。
* **直觉：** LN 就像是**只看"小明"这一个同学**，把他自己的数学、英语、物理成绩放在一起计算均值和方差来标准化。它不关心其他同学考得怎么样，只关心这个样本内部特征的相对分布。
* **适用场景：** 自然语言处理 (RNN, Transformer/BERT/GPT)。

## 2. 深度解析

### Batch Normalization (BN)

**计算方式：**

对于第 $c$ 个通道，计算 Batch 中所有样本在该通道上的均值 $\mu_c$ 和方差 $\sigma_c^2$：
$$\hat{x}_{n,c} = \frac{x_{n,c} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

然后引入可学习的参数 $\gamma$ (缩放) 和 $\beta$ (平移) 来恢复表达能力：
$$y_{n,c} = \gamma_c \hat{x}_{n,c} + \beta_c$$

**优点：**
* 极大地加速了 CNN 的训练
* 具有一定的正则化效果（因为 Batch 统计量引入了噪声），减少了对 Dropout 的依赖

**缺点：**
* **高度依赖 Batch Size：** 如果 Batch Size 太小（例如 < 8），统计出的均值和方差不准确，导致模型性能急剧下降
* **RNN/序列数据难处理：** RNN 处理的序列长度不一，不同时间步使用不同的统计量非常麻烦且效果不佳
* **训练与推理不一致：** 训练时用当前 Batch 的统计量，推理（测试）时需要使用训练期间积累的"移动平均"统计量

### Layer Normalization (LN)

**计算方式：**

对于第 $n$ 个样本，计算该样本所有特征（通道）的均值 $\mu_n$ 和方差 $\sigma_n^2$：
$$\hat{x}_{n,c} = \frac{x_{n,c} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}}$$

同样也有可学习参数 $\gamma$ 和 $\beta$（针对每个元素或通道）。

**优点：**
* **独立于 Batch Size：** 即使 Batch Size = 1 也能正常工作
* **适合序列模型：** 在 NLP 中（如 Transformer），不同样本的句子长度不同，特征含义也不同，LN 针对单个样本独立计算，非常适合这种变长数据
* **训练推理一致：** 训练和测试时计算方式完全相同，不需要维护移动平均

**缺点：**
* 在 CNN（图像任务）中，效果通常不如 BN。因为图像的不同通道（如 RGB 或特征图）往往代表不同的视觉特征，把它们混在一起计算均值有时会破坏信息的独立性

## 3. 总结对照表

| 特性 | Batch Normalization (BN) | Layer Normalization (LN) |
| :--- | :--- | :--- |
| **归一化维度** | 跨样本 (Batch dimension) | 跨特征 (Feature/Channel dimension) |
| **Batch Size 依赖** | **强依赖** (小 Batch 效果差) | **无依赖** (不影响) |
| **计算统计量对象** | 同一特征，不同样本 | 同一样本，不同特征 |
| **主要应用领域** | **CV (图像分类, 目标检测)** | **NLP (Transformer, RNN)** |
| **训练/推理模式** | 不同 (推理需用移动平均) | 相同 |
| **直观理解** | 比较全班同学的"单科成绩" | 比较某个同学的"各科综合表现" |

## 4. Transformer 中的层归一化

在 NLP（自然语言处理）和 Transformer 中，数据通常是这样的：

> 假设我们在处理一句话："**我爱吃**"。

这句话有 3 个字（Token）。在计算机里，每个字都是一个**向量**（一串数字）。假设每个字用 4 个数字来表示（实际上是几百到几千个）：

* **我：** `[10, 20, 999, -5]`  ← 注意那个 999，数值特别大
* **爱：** `[0.1, 0.2, 0.1, 0.1]`
* **吃：** `[5, 5, 5, 5]`

### 如果没有 LN 会发生什么？

"我"这个字里面有一个 `999`，它的数值远大于"爱"和"吃"。如果不处理，只要数据一进神经网络，那个 `999` 就会像个大喇叭一样，掩盖掉其他所有细微的信息。神经网络会只盯着那个大数看，学不到东西。

### LN 是怎么做的？

LN 这个"整理师"会**分别**走到每一个字面前，对它们说："你们每个人，都把自己内部的数字整理一下，不要太夸张。"

* **对"我"做 LN：** 算出 `[10, 20, 999, -5]` 的均值和方差，把它们压缩回一个标准的范围（比如 -1 到 1 之间）。那个 `999` 被拉下来了
* **对"爱"做 LN：** 算出 `[0.1, 0.2, 0.1, 0.1]` 的均值和方差，把它拉伸到一个正常的范围

**关键点：** "我"的归一化，完全不看"爱"和"吃"的数据。**每个人（每个 Token）只管自家的事**，这就是"层"归一化。

## 5. LN 与注意力机制的结合

在 Transformer 中，LN 和注意力（Self-Attention）通常是像**三明治**一样夹在一起的。主要有两种夹法：

### Post-LN（原始 Transformer 的做法）

```
输入 → Attention → Add (残差) → LN → FFN → Add (残差) → LN → 输出
```

流程：
1. **输入进来**（比如"我"的向量）
2. **先过注意力机制**（Attention）：这步是让"我"去看看"爱"和"吃"，融合上下文信息
3. **残差连接**（Residual）：把"原本的我不变的向量" + "注意力处理后的新向量"加起来
4. **最后做 LN**：对加起来的结果进行归一化

> **问题：** 这种做法在层数很深的时候（比如 100 层），梯度很容易消失，模型很难训练。

### Pre-LN（GPT、Llama 等大模型的做法）

```
输入 → LN → Attention → Add (残差) → LN → FFN → Add (残差) → 输出
```

流程：
1. **输入进来**
2. **先做 LN**：先把"我"的向量整理干净，去掉极端数值
3. **再过注意力机制**：用整理好的干净数据去算注意力
4. **残差连接**：把结果加回去

> **优势：** 训练更稳定，是目前大模型的主流做法。

## 6. 为什么注意力机制需要配合 LN？

注意力机制的核心公式：
$$\text{Softmax} \left( \frac{Q \times K^T}{\sqrt{d}} \right)$$

这里有两个关键步骤容易出问题，必须靠 LN 来救：

### 防止"点积"爆炸

注意力机制的核心是计算 $Q$ 和 $K$ 的**点积**（也就是相似度）。

如果输入的向量里有 `999` 这种大数：
* 点积算出来的结果会巨大无比
* 这就好比大家投票，"我"这一票直接投了 100 万分，别人的票都是 1 分

### 拯救 Softmax

Softmax 是一个"赢家通吃"的函数：
* 如果点积结果都很小（比如 1.1, 1.2, 0.9），Softmax 之后大家都有概率，网络能学到细腻的关系
* 如果点积结果巨大（比如 1000, 5, 2），Softmax 之后，那个 1000 对应的概率就是 **100%**，其他全是 **0%**
* **后果：** 梯度变成了 0，反向传播推不动，模型直接"死"了，根本训练不起来

**LN 的作用就是：** 在进 Attention 之前（Pre-LN）或者之后（Post-LN），把数据强行拉回一个 0 均值、1 方差的稳定区间，保证 Softmax 能够正常工作，算出有效的梯度。

## 7. 什么时候用哪个？

* **如果你在做图像处理 (CNN, ResNet, EfficientNet)：** 首选 **BN**。除非你的显存受限导致 Batch Size 只能开到 1 或 2，这时可以考虑 Group Normalization (GN) 或 LN
* **如果你在做自然语言处理 (Transformer, BERT, GPT, LSTM)：** 首选 **LN**。BN 在文本数据上表现通常很差
* **如果你在做生成对抗网络 (GAN)：** 这是一个特例，有时会用到 Instance Normalization (IN) 或 Spectral Normalization，但在某些层也会用到 BN

## 8. 形象类比总结

如果把训练模型比作大家一起**做广播体操**：

1. **Batch Norm (BN)** 是教导主任喊："全班同学注意，现在按**身高**排队，最高和最矮的都要调整一下！"（这在 NLP 里行不通，因为每句话长短不一样，含义也不一样，不能强行按全班比）

2. **Layer Norm (LN)** 是每个人自己的**私人教练**：
   * 教练对你说："别管别人，你今天状态太亢奋了（数值太大），深呼吸冷静一下（归一化）。"
   * 教练对他说："你今天太低沉了（数值太小），兴奋一点（归一化）。"

**结合注意力机制：** 就是为了保证你在做**最难的动作（Self-Attention）**之前，你的身体状态是**标准、健康**的。如果你处于极度亢奋或极度低沉的状态去练高难度动作，要么动作变形（学不到特征），要么直接受伤（梯度消失/爆炸）。

## 9. PyTorch 代码示例

```python
import torch
import torch.nn as nn

# 创建示例数据: [batch_size, seq_len, hidden_dim]
batch_size, seq_len, hidden_dim = 2, 3, 4
x = torch.randn(batch_size, seq_len, hidden_dim)

# Layer Normalization (用于 Transformer)
ln = nn.LayerNorm(hidden_dim)
ln_output = ln(x)  # 对每个 token 的 hidden_dim 维度归一化

# Batch Normalization (用于 CNN)
# 注意：需要调整维度 [batch, channel, height, width]
x_cnn = torch.randn(2, 64, 32, 32)  # 2 张图片, 64 通道, 32x32
bn = nn.BatchNorm2d(64)
bn_output = bn(x_cnn)  # 对每个通道跨 batch 归一化

print(f"LN 输入形状: {x.shape}, 输出形状: {ln_output.shape}")
print(f"BN 输入形状: {x_cnn.shape}, 输出形状: {bn_output.shape}")
```

## 参考资料

- [Layer Normalization 原始论文 (2016)](https://arxiv.org/abs/1607.06450)
- [Batch Normalization 原始论文 (2015)](https://arxiv.org/abs/1502.03167)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
