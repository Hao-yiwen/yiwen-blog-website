---
title: NVIDIA 常见数值格式速查
sidebar_label: NVIDIA 数值格式
sidebar_position: 1
date: 2025-11-13
last_update:
  date: 2025-11-13
---

# NVIDIA 常见数值格式速查：FP16 / BF16 / TF32 / INT8 / INT4

## 1. 背景：这些格式都是干嘛的？

在 NVIDIA GPU 上，大部分 AI 计算跑在 **Tensor Core** 上。
Tensor Core 支持多种精度（FP16、BF16、TF32、INT8、INT4…），用来**在精度和性能之间做取舍**。

粗略来说：

* **FP 系列（浮点）**：适合训练，数值范围大，对梯度比较友好。
* **INT 系列（整数）**：适合量化推理，超省显存、带宽，性能高，但精度要靠量化技巧兜底。

---

## 2. 各种格式的区别 & 何时开始支持

### 2.1 FP16（半精度浮点）

* **位宽/结构**：16 bit 浮点（1 符号 + 5 指数 + 10 尾数），动态范围比 FP32 小，精度比 BF16 更高。
* **特点**：

  * 比 FP32 少一半存储/带宽；
  * 在可表示范围内精度不错，但容易溢出/下溢。
* **典型用途**：

  * 早期混合精度训练主力（AMP）；
  * 现在仍广泛用于训练 / 推理（尤其是没用 BF16 的场景）。
* **NVIDIA 首次 Tensor Core 支持**：

  * **Volta 架构（V100）**：第一代 Tensor Core，只支持 FP16 矩阵乘。

---

### 2.2 BF16（bfloat16）

* **位宽/结构**：16 bit 浮点（1 符号 + 8 指数 + 7 尾数）。

  * 指数位和 FP32 一样 → **动态范围几乎等于 FP32**；
  * 尾数变少 → 精度比 FP16 低一些。
* **特点**：

  * 不容易溢出/下溢，训练大模型更稳；
  * 和 FP32 转换简单（砍尾数即可）。
* **典型用途**：

  * 现在大模型训练非常常用（尤其 A100/H100/B100 这种卡）；
  * 常见模式：**参数/激活 BF16，累加/部分层用 FP32 或 BF16 高精度**。
* **NVIDIA 首次 Tensor Core 支持**：

  * **Ampere 架构（A100）**：第三代 Tensor Core 加入 BF16 支持。

---

### 2.3 TF32（TensorFloat-32）

* **本质**：不是一个"存储 dtype"，而是 **Tensor Core 上的一种计算模式**。

  * 大致：保留 FP32 的 8 位指数，用类似 FP16 的 10 位尾数，在硬件里做截断。
* **特点**：

  * 代码里仍然用 FP32 存储；
  * Tensor Core 内部把 FP32 运算"偷偷降精度"为 TF32 来加速；
  * 对大多数网络，几乎不用改代码就能比纯 FP32 快很多。
* **典型用途**：

  * 用 FP32 写的老代码，直接在 A100/H100 上获得加速；
  * 训练中"懒得改精度"的场景：直接启用 TF32 获得 Tensor Core 加速。
* **NVIDIA 首次支持**：

  * **Ampere 架构（A100）**：第三代 Tensor Core 新增 TF32 模式。

---

### 2.4 INT8（8 位整数）

* **位宽/结构**：8 bit 整数，一般配合一个 scale（和可能的 zero-point）一起使用。
* **特点**：

  * 超省显存、带宽，吞吐量高；
  * 需要量化（把 FP32/BF16 的值映射到 INT8），有精度损失；
  * 对于推理来说，INT8 已经是非常成熟的量化方案。
* **典型用途**：

  * 推理量化（CNN / LLM 都很多）；
  * 少数训练研究方案中做权重/激活低精度存储。
* **NVIDIA 首次 Tensor Core 支持**：

  * **Turing 架构（T4、RTX 20 系列）**：第二代 Tensor Core 新增 INT8、INT4 等多精度支持。
  * Ampere / Hopper / Blackwell 继续支持并提升性能。

---

### 2.5 INT4（4 位整数）

* **位宽/结构**：4 bit 整数，同样需要量化 scale。
* **特点**：

  * 存储、带宽只有 FP16 的 1/4，非常适合极致压缩；
  * 精度更差，对量化算法要求高（per-channel scale、outlier 处理等）；
  * 更多用于 **推理**，尤其是大型模型的 4bit 量化（GPTQ、AWQ、QLoRA 等）。
* **典型用途**：

  * 大模型 4bit 量化推理；
  * 低精度微调（QLoRA 这类）。
* **NVIDIA 首次 Tensor Core 支持**：

  * 同样是 **Turing 架构** 的第二代 Tensor Core 加入 INT4 支持，后续架构延续并优化。

---

## 3. 时间线速查表（按 NVIDIA 架构）

> 这里只列和深度学习相关、常见的几代数据中心/专业 GPU。

| 架构 & 代表卡                                 | Tensor Core 代数 | 首次/主要支持的精度（只写和你关心的相关）                                                          |
| ---------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| **Pascal（P100 等）**                       | 无 Tensor Core  | 只支持 FP32/FP64 常规计算（无 FP16 Tensor Core）                                         |
| **Volta（V100）**                          | 第 1 代          | **FP16 Tensor Core**（训练混合精度起点）                             |
| **Turing（T4, RTX 20）**                   | 第 2 代          | 在 Volta FP16 基础上新增 **INT8 / INT4** Tensor Core 精度                 |
| **Ampere（A100, A30, 部分 RTX A 系列）**       | 第 3 代          | 保留 **FP16 + INT8/INT4**，新加 **BF16**、**TF32** 支持                   |
| **Hopper（H100/H200）**                    | 第 4 代          | 延续 **FP16/BF16/TF32/INT8/INT4**，新加 **FP8**（两种 nvFP8 格式） |
| **Blackwell（B100/GB200, RTX Blackwell）** | 第 5 代          | 延续上面所有，并加强 **FP8**，新增 **FP4** Tensor Core 精度（极致低精度）         |

---

## 参考资料

- [NVIDIA Tensor Core Architecture](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
