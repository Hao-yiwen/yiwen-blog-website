---
title: Vision Transformer (ViT) 深度解析
sidebar_label: Vision Transformer
date: 2025-12-16
last_update:
  date: 2025-12-16
tags: [transformer, vision, vit, image-classification, deep-learning]
---

# Vision Transformer (ViT) 深度解析

**对应论文：** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR 2021)*

**核心思想：** 彻底抛弃卷积神经网络 (CNN) 的归纳偏置，将图像完全视为通过"切块"得到的序列，直接使用标准的 Transformer 架构进行图像分类。

---

## 1. 核心架构流程 (Architecture Pipeline)

ViT 的处理流程可以概括为："切块 → 铺平 → 线性投影 → 增加位置信息 → Transformer 编码 → 分类"。

### 第一步：图像切块 (Patching)

* **输入：** 一张二维图像 $x \in \mathbb{R}^{H \times W \times C}$（例如 $224 \times 224 \times 3$）。
* **操作：** 将图像切分成固定大小的 $N$ 个方块 (Patches)。
* **参数：** 假设 Patch 大小为 $P \times P$（通常是 $16 \times 16$）。
* **结果：** 图像变成了一个序列，序列长度 $N = \frac{H \cdot W}{P^2}$。

### 第二步：线性投影 (Linear Projection) —— 也就是 "Embedding"

这是 ViT 能够处理像素的关键一步。

* **操作：** 将每个 $P \times P \times C$ 的 Patch **展平 (Flatten)** 为一个一维向量。
* **映射：** 通过一个全连接层（Linear Layer），将这个扁平向量映射到模型内部的隐藏层维度 $D$。
* **数学表示：**

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}}
$$

其中 $\mathbf{E}$ 就是这个线性投影矩阵。

* **注：** 正如之前讨论的，代码实现中通常使用 `Conv2d(kernel_size=16, stride=16)` 来一步完成切块和投影。

### 第三步：特殊标记与位置编码

Transformer 本身不具备空间感知能力，也不懂分类任务，因此需要两个辅助组件：

**1. [CLS] Token (分类标记)：**

* 借鉴 BERT，在序列的最前面人为插入一个**可学习的向量**。
* 在经过所有 Transformer 层后，模型只提取这个 Token 的输出来代表整张图的特征，用于分类。

**2. Position Embedding (位置编码)：**

* 因为 Transformer 把 Patch 视为无序的集合，为了让模型知道"哪个块在左上，哪个在右下"，必须加上位置信息。
* ViT 使用**可学习的一维位置编码**，直接加（Add）到 Patch Embedding 上。

### 第四步：Transformer Encoder

这是模型的主体，完全沿用 NLP 中的标准架构：

* **MSA (Multi-head Self Attention)：** 多头自注意力机制，负责计算 Patch 之间的全局关联（例如，让左上角的"猫耳朵"关注到右下角的"猫尾巴"）。
* **MLP (Multi-Layer Perceptron)：** 前馈神经网络，用于特征变换。
* **LayerNorm (LN)：** 层归一化，放在每个块的前面 (Pre-Norm)。
* **Residual Connection：** 残差连接。

---

## 2. 为什么 ViT 能成功？(原理分析)

### 2.1 归纳偏置 (Inductive Bias) 的权衡

这是 ViT 与 CNN 最大的区别：

| 特性 | CNN (ResNet) | Transformer (ViT) |
| --- | --- | --- |
| **归纳偏置** | **强** (Strong) | **弱** (Weak) |
| **平移不变性** | 自带 (Translation Invariance) | 需要学习 |
| **局部性** | 自带 (Locality, 只看局部窗口) | 需要学习 (Attention 是全局的) |
| **数据需求** | 小数据也能训练 (因为有先验知识) | **极大** (需要海量数据来自己学会这些规则) |

* **结论：** CNN 像是"带着偏见（先验知识）"看图，学得快但上限受限；ViT 是一张白纸，早期学得慢，但如果有足够的数据（如 JFT-300M）让它自己建立对世界的理解，它的上限极高。

### 2.2 感受野 (Receptive Field)：CNN vs ViT

这是 ViT 与 CNN 最反直觉的区别。

**CNN 的逻辑：坐井观天，慢慢爬升**

* **第一层 (Local)：** 卷积核（如 $3 \times 3$）就像小窗口，只能看到局部的 9 个像素。
* **中间层 (Expanding)：** 随着层数加深和池化操作，视野逐渐扩大。
* **最后几层 (Global)：** 只有到了最深处，神经元才能通过层层传递"看"到整张图。

**ViT 的逻辑：上帝视角，一步到位**

* **机制：** 自注意力机制没有"距离"概念。每个 Patch 都与**所有其他 Patch** 直接计算相似度。
* **第一层 (Global)：** 数据进入第 1 层时，左上角的 Patch 就已经和右下角的 Patch 直接交互了。

**比喻：**
- CNN 像传话游戏：第 1 个人只能跟身边的人说话，到第 20 个人才知道整个队伍的信息。
- ViT 像微信群：一发消息，所有人瞬间都能看到。

**实际观察：** 虽然 ViT 有能力在第一层看全局，但训练后的模型第一层往往仍倾向于关注局部（类似卷积）。不过确实有一些 Attention Head 在第一层就关注整张图。

| 架构 | 第一层感受野 | 全局感受野 |
|------|-------------|-----------|
| CNN | 局部（物理限制） | 需堆叠多层 |
| ViT | 全局（可选） | 第一层即可 |

### 2.3 扩展能力 (Scalability)

根据 Scaling Law 的分析：

* **数据扩展：** 随着训练数据量达到亿级（$10^8$ 以上），ViT 的性能曲线没有饱和迹象，反超 ResNet。
* **计算效率：** 在大算力预算下，ViT 的计算效率（Compute Efficiency）优于 CNN。同样的算力，ViT 能换来更高的准确率。

---

## 3. 训练方式：监督 vs 自监督

ViT 的训练方式经历了演变：

### 3.1 监督预训练 (Supervised Pre-training) —— 原论文的核心

* **方法：** 使用带有标签的超大数据集（如 ImageNet-21k 或 JFT-300M）进行分类训练。
* **结果：** 这是 ViT 击败 ResNet 的关键。必须先在大数据集上预训练，然后再迁移到小数据集（ImageNet-1k, CIFAR）上微调。
* **局限：** 极其依赖昂贵的标注数据。

### 3.2 自监督预训练 (Self-supervised) —— 探索与未来

**Masked Patch Prediction:** 类似于 BERT 的完形填空。

* 做法：随机挖掉一些 Patch，让模型预测被挖掉部分的像素值（或平均颜色）。
* 表现：在原论文中，这种方法比从头训练好，但不如有监督训练。

**后续发展 (MAE, BEiT)：** 在 ViT 论文发表后，何恺明等人的 **MAE (Masked Autoencoders)** 改进了这一步，证明了**掩码自监督**在视觉领域其实可以达到甚至超越监督学习的效果。

---

## 4. 可视化解释：ViT 到底学到了什么？

当我们查看 Linear Projection 层的权重时：

* **自动习得卷积核：** ViT 自动学会了类似于 Gabor 滤波器的纹理检测功能和颜色斑点检测功能。
* **这意味着：** 即使架构里没有卷积层，只要数据足够多，优化器也会逼迫线性层进化出处理视觉底层特征（边缘、颜色）的能力。

---

## 5. 总结：ViT 的优缺点

### 优点

1. **全局视野 (Global Receptive Field)：** 从第一层开始就能看到整张图，不像 CNN 需要堆叠很多层才能看到全局。
2. **上限极高：** 数据越多，效果越好，非常适合如今的大模型时代。
3. **多模态统一：** 图片和文字都被变成了 Token 序列，这为后来的多模态模型（如 Gemini, GPT-4V）奠定了统一的架构基础。

### 缺点

1. **数据饥渴：** 在小数据集（如只有几千张图）上，效果通常不如 ResNet。
2. **计算复杂度：** 对高分辨率图像，Attention 的计算量是序列长度的平方级 $O(N^2)$ 增长，导致处理大图非常慢（后来出现了 Swin Transformer 等变体来解决这个问题）。

---

## 6. 参考资料

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - ViT 原始论文
- [The Illustrated ViT](https://jalammar.github.io/illustrated-vit/) - 图解 ViT
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) - MAE 论文
- [Swin Transformer](https://arxiv.org/abs/2103.14030) - 解决 ViT 计算复杂度问题的变体
