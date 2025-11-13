---
sidebar_position: 2
---

# NVIDIA GPU 算力对比表：FP32 / FP16 / Tensor Core 全览

## 说明

* **FP32 / FP16（非 Tensor）**：普通 CUDA 浮点算力。对大部分 GeForce / RTX 工作站卡来说，FP16 标称带宽 ≈ FP32，所以直接写成相同数值。
* **FP16 Tensor**：Tensor Core 的 FP16/BF16 理论峰值，更接近你训练/推理时真正关心的算力。部分 Hopper/Blackwell 官方数据是 *带稀疏* 的峰值，dense 一般约为 1/2，在备注里说明。
* 数值都取自官方 datasheet 或大厂方案商的规格表，有的四舍五入到 0.1 TFLOPS 左右。

---

## 1️⃣ 消费级 GeForce RTX 40 / 50 系列

> 单位：TFLOPS；FP16（非 Tensor）基本可视为 = FP32。

| 型号              | 架构                | FP32                            | FP16（非 Tensor） | FP16 Tensor（大致，dense）                                        | 显存                               | 备注            |
| --------------- | ----------------- | ------------------------------- | -------------- | ------------------------------------------------------------ | -------------------------------- | ------------- |
| RTX 5090        | Blackwell (GB202) | **104.8**           | ≈104.8         | ≈**419** FP16 Tensor（≈838 含稀疏）                 | 32 GB GDDR7          | 顶级消费卡，50 系列旗舰 |
| RTX 5080        | Blackwell         | **56.3**            | ≈56.3          | ≈**142.3** FP16 Tensor（≈284.6 含稀疏）       | 16 GB GDDR7        | 高端 4K/AI 卡    |
| RTX 5070 Ti     | Blackwell         | **43.9**            | ≈43.9          | 官方暂未单独给出                                                     | 16 GB GDDR7    | 中高端           |
| RTX 5070        | Blackwell         | **30.9**            | ≈30.9          | 官方暂未单独给出                                                     | 12 GB GDDR7           | 主流 2K         |
| RTX 5060 Ti     | Blackwell         | **23.7**            | ≈23.7          | —                                                            | 8 或 16 GB GDDR7（不同板卡） | 主流卡           |
| RTX 5060        | Blackwell         | **19.2**            | ≈19.2          | —                                                            | 8 GB GDDR7           | 入门 2K/1080p   |
| RTX 5050        | Blackwell         | **13.2**            | ≈13.2          | —                                                            | 8 GB GDDR6           | 入门级           |
| RTX 4090        | Ada (AD102)       | **82.6**                        | ≈82.6          | **≈330** FP16 Tensor（≈661 含稀疏）           | 24 GB GDDR6X         | 40 系列旗舰，AI 常用 |
| RTX 4080 SUPER  | Ada               | **52.2**         | ≈52.2          | ≈**418** FP16 Tensor（≈836 含稀疏）         | 16 GB GDDR6X      | 高端            |
| RTX 4070        | Ada               | **29.1** | ≈29.1          | ≈**233** FP16 Tensor（≈466 含稀疏）         | 12 GB GDDR6X                     | 主流 2K 游戏/轻量训练 |
| RTX 4060 Ti 8GB | Ada               | **22.1**         | ≈22.1          | ≈**177** FP16 Tensor（估算，参考 NVIDIA/评测表） | 8 GB GDDR6                       | 性价比中端，显存偏小    |

---

## 2️⃣ RTX 工作站 / L 系列

> 这些更偏专业渲染 / 推理 / 轻量训练。

| 型号                     | 定位        | 架构        | FP32                                         | FP16（非 Tensor） | FP16 Tensor（dense 近似）                                       | 显存                                 | 备注                |
| ---------------------- | --------- | --------- | -------------------------------------------- | -------------- | ----------------------------------------------------------- | ---------------------------------- | ----------------- |
| RTX 6000 Ada           | 工作站       | Ada       | **≈91.1** TFLOPS          | ≈91.1          | ≈**165** TFLOPS FP16 Tensor（≈330 含稀疏）    | 48 GB GDDR6 ECC | 很常见的本地训练/推理卡      |
| RTX Pro 6000 Blackwell | 工作站       | Blackwell | **≈109.7** TFLOPS FP32        | ≈109.7         | ≈**503.8** TFLOPS FP16 Tensor（≈1007.6 含稀疏）    | 96 GB GDDR7      | 新一代高端工作站/服务器卡     |
| L20                    | 数据中心推理    | Ada       | **≈59.4** TFLOPS FP32 | ≈59.4          | **≈59.35** TFLOPS FP16 Tensor（dense）       | 48 GB GDDR6     | 低功耗推理，性价比高        |
| L40S                   | 数据中心训练/推理 | Ada       | **≈91.6** TFLOPS FP32      | ≈91.6          | **≈362.1** TFLOPS FP16 Tensor（≈733 含稀疏） | 48 GB GDDR6      | 很接近 A100 级别的 AI 卡 |

---

## 3️⃣ 数据中心训练卡（V100 / A100 / H100 / H200 / H20 / GB200 / GB300）

> 下面的 FP16 基本都是 **Tensor Core 算力**，更接近实际训练/推理速度。

| 型号                 | 架构              | FP32                                                                   | FP16 Tensor（标称）                                                                                        | 显存                                                                  | 备注                                               |
| ------------------ | --------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------ |
| Tesla V100 32GB    | Volta           | ≈**14.1** TFLOPS FP32                                  | ≈**125** TFLOPS（SXM，Tensor Core；PCIe 约 112）                                       | 32 GB HBM2                                           | 老牌训练卡，很多老集群还在用                                   |
| A100 80GB          | Ampere          | **19.5** TFLOPS FP32                                 | **312** TFLOPS FP16 Tensor（**624** 含稀疏）                                           | 80 GB HBM2e，≈2 TB/s                                  | Hopper 前一代主力                                     |
| H100 80GB SXM      | Hopper          | **67** TFLOPS FP32                                      | 数据手册：**1,979 TFLOPS FP16 Tensor（含稀疏）**，dense ≈**990** TFLOPS                             | 80 GB HBM3，3.35 TB/s                                 | 现在很多云厂商的主力训练卡                                    |
| H200 SXM           | Hopper+         | **67** TFLOPS FP32                                       | 官方/合作伙伴给出：**1,979 TFLOPS FP16 Tensor**（基本同 H100，dense 约 990）                              | 141 GB HBM3e，≈4.8 TB/s                                | 相当于「大显存版 H100」                                   |
| H20 96GB           | Hopper（中国阉割版）   | （FP32 官方没太强调，约 60+）                              | 资料里给：**≈148 TFLOPS FP16/BF16 Tensor（dense）**，INT8/FP8≈296 TOPS/TFLOPS            | 96 GB HBM3                                   | 针对受限市场的「降配 H100」                                 |
| GB200 **（每颗 GPU）** | Blackwell       | ≈**80** TFLOPS FP32（Blackwell Ultra/B200 级） | ≈**5 PFLOPS** FP16/BF16 Tensor（dense），≈10 PFLOPS 含稀疏（NVL72 或 superchip 汇总时） | ≈192–288 GB HBM3e（随具体 B200/GB200 版本略有差异）        | GB200 一般以 Grace+2×Blackwell 组成 superchip 使用      |
| GB300 **（每颗 GPU）** | Blackwell Ultra | ≈**80** TFLOPS FP32（GB300 Ultra）            | NVL72 机柜总计 **360 PFLOPS FP16**，72 GPU → 每 GPU ≈**5 PFLOPS FP16**（dense）                  | 单 GPU 约 279 GB HBM3e，8 TB/s 带宽（Ultra 版本） | GB300 相比 GB200 FP16 提升约 1.5×，更偏推理/"AI reasoning" |

---

## 解读建议

### 关于稀疏性

* Hopper/Blackwell 的官方 FP16/BF16 数经常写的是「**带结构化稀疏**」的峰值（比如 H100 的 1,979 TFLOPS），**dense 算力大约除以 2**。
* 对 GeForce / RTX 6000 Ada 这类卡，如果你只用普通 CUDA FP16，而不用 Tensor Core，FP16 算力基本就按 FP32 那一列来估算。

### 显存带宽

* 显存带宽对训练/推理性能影响很大，HBM3/HBM3e 的带宽远超 GDDR6/GDDR6X/GDDR7
* 数据中心卡通常配备更高的显存带宽（2-8 TB/s），消费级卡通常在 0.5-1 TB/s 范围

### 功耗考虑

* 数据中心卡功耗通常在 300-700W 范围
* 消费级卡功耗通常在 200-450W 范围
* 选择 GPU 时需要考虑数据中心的供电和散热能力

---

## 参考资料

- [NVIDIA Official GPU Specifications](https://www.nvidia.com/en-us/data-center/resources/datasheets/)
- [NVIDIA Tensor Core Architecture](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [GPU Performance Benchmarks](https://bizon-tech.com/gpu-benchmarks/)
- [Wikipedia: GeForce RTX 40 series](https://en.wikipedia.org/wiki/GeForce_RTX_40_series)
- [Wikipedia: GeForce RTX 50 series](https://en.wikipedia.org/wiki/GeForce_RTX_50_series)
