---
title: 当前主流深度学习架构深度解析
sidebar_position: 26
tags: [深度学习, Transformer, Diffusion, DiT, AIGC, LLM, 文生图, 视频生成]
---

# 当前主流深度学习架构深度解析

当前深度学习领域，尤其是生成式 AI（AIGC）方向，最主流的架构可以概括为 **"Transformer 统治一切，Diffusion 处理多模态"**，并且两者正在呈现明显的 **融合趋势**。

以下是针对文本、图像和视频生成三大领域的当前主流架构深度解析：

---

## 1. 文本生成 (LLMs)

**核心架构：Transformer (Decoder-only)**

这是目前几乎所有大语言模型（LLM）的基石。虽然 Google 最早提出的 Transformer 包含 Encoder 和 Decoder，但现在的生成式模型几乎都只使用 **Decoder-only** 架构。

### 主流技术细节

-   **Self-Attention (自注意力机制)：** 核心灵魂，让模型能够理解上下文的长距离依赖关系。
-   **MoE (Mixture of Experts，混合专家模型)：** 这是当下最火的"提效"架构（如 GPT-4, Mixtral, DeepSeek）。它不再每次激活所有参数，而是根据问题激活一小部分"专家"网络。这使得模型参数量可以做得极大（如万亿级），但推理成本却较低。
-   **长上下文 (Long Context)：** 通过 RoPE (旋转位置编码) 等技术，让模型能处理 100k 甚至 1M token 的超长文本（如 Claude 3, Kimi）。

:::tip 代表模型
GPT-4, Llama 3, Claude 3, Qwen (通义千问)
:::

---

## 2. 文生图 (Text-to-Image)

**核心架构：Diffusion Model (扩散模型)**

图像生成领域经历了从 GAN (生成对抗网络) 到 Diffusion 的彻底范式转移。目前的扩散模型架构正在发生代际更替：

### 上一代主流：UNet + Diffusion

-   早期和中期的 Stable Diffusion (如 SD 1.5, SDXL) 使用 **UNet** 作为核心网络来预测噪声。
-   UNet 擅长处理像素级的细节，但扩展性（Scaling）不如 Transformer。

### 当前最前沿：DiT (Diffusion Transformer)

-   这是现在的"当红炸子鸡"。它将 UNet 替换为 **Transformer**。
-   **原理：** 将图片切成一个个小块（Patches），像处理文本 Token 一样处理图片块。
-   **优势：** 极佳的扩展性（模型越大效果越好），对语义的理解能力更强，生成的文字和复杂构图更准确。

:::tip 代表模型

**UNet 派：** Stable Diffusion 1.5/XL, Midjourney v5

**DiT 派 (最新趋势)：** **Flux.1** (目前开源界最强), Stable Diffusion 3, Midjourney v6 (推测), DALL-E 3
:::

---

## 3. 视频生成 (Video Generation)

**核心架构：DiT (Diffusion Transformer) + Spacetime Patches**

视频生成在 2024 年爆发（以 Sora 为标志），其核心架构基本完全统一到了 **DiT** 上。

### 核心逻辑

-   视频被视为"三维"数据（高度、宽度、时间）。
-   模型将视频在空间和时间上切分成 **Spacetime Patches (时空补丁)**。
-   使用 Transformer 全局处理这些补丁，这使得模型不仅能生成高质量画面，还能保持几秒甚至一分钟内的**连贯性**（物体不会变形、物理规律基本正确）。
-   **3D VAE：** 用于将视频压缩到潜空间（Latent Space）进行处理，再还原成像素视频。

:::tip 代表模型
OpenAI **Sora**, 快手 **可灵 (Kling)**, Luma **Dream Machine**, Runway **Gen-3 Alpha**
:::

---

## 总结：架构的大一统趋势

如果看一张图表，你会发现所有领域都在向 **Transformer** 收敛：

| 领域     | 过去的主流      | 现在的绝对主流              | 趋势                                   |
| :------- | :-------------- | :-------------------------- | :------------------------------------- |
| **文本** | RNN / LSTM      | **Transformer (Decoder)**   | MoE 架构普及，追求超长上下文           |
| **图像** | GAN / CNN       | **Diffusion (UNet)**        | **Diffusion (Transformer / DiT)**      |
| **视频** | 3D-CNN          | **DiT (时空 Transformer)**  | 能够理解物理规律的世界模型             |

### 一句话概括

现在的 AI 实际上大多是 **"穿着不同外衣的 Transformer"**。无论是处理文字、生成图片还是渲染视频，核心都是通过 Transformer 的注意力机制来学习数据之间的关联。
