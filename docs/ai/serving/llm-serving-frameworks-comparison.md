---
title: vLLM vs TorchServe vs Triton：LLM 推理框架对比
sidebar_position: 1
tags: [vLLM, TorchServe, Triton, 推理, 部署, LLM]
---

# vLLM vs TorchServe vs Triton：LLM 推理框架对比

## 引言

在部署大语言模型（LLM）时，vLLM、TorchServe 和 NVIDIA Triton 这三个推理框架经常被提及。虽然它们都能把模型部署成 API 服务，但它们的设计理念和适用场景却大不相同。

**核心区别：通用性 vs 专精性**

简单类比：
- **TorchServe & Triton**：综合医院（什么病都看，内科外科儿科都行）
- **vLLM**：心脏专科医院（只看心脏病，但在这个领域是世界第一）

---

## 一、核心区别：处理的模型类型不同

### TorchServe / NVIDIA Triton（通用型）

**特点：万金油框架**

- **支持范围广**：可以运行任何深度学习模型
  - 图像分类（ResNet、EfficientNet）
  - 目标检测（YOLO、Faster R-CNN）
  - 语音识别（Whisper、Wav2Vec）
  - 文本分类（BERT、RoBERTa）
  - 推荐系统（DeepFM、Wide&Deep）

- **工作原理**：
  ```
  输入数据 → 模型黑盒 → 输出结果
  ```
  它们将模型视为黑盒，不关心内部细节，只负责：
  1. 加载模型权重
  2. 接收请求
  3. 执行前向传播
  4. 返回结果

### vLLM（大模型专用型）

**特点：LLM 极致优化**

- **支持范围窄**：只能运行 Transformer Decoder 架构的生成式 LLM
  - ✅ 支持：Llama、Qwen、Mistral、GPT-2/3、Baichuan
  - ❌ 不支持：YOLO（图像）、Whisper（语音）、BERT（编码器模型）

- **工作原理**：
  ```
  用户输入 → 自回归生成 → 一个 token 一个 token 输出
  ```
  深入优化 LLM 特有的生成过程：
  - 理解 Attention 层的计算模式
  - 专门优化 KV Cache 管理
  - 针对"生成下一个词"的场景设计

:::tip 为什么 vLLM 不支持其他模型？
因为它的核心技术（PagedAttention、Continuous Batching）都是为自回归生成过程量身定制的。如果要支持图像模型，就失去了专精的意义。
:::

---

## 二、核心区别：Batching（批处理）机制

这是 **vLLM 存在的核心意义**，也是它在 LLM 推理上吊打通用框架的地方。

### 场景设定

假设有两个用户同时发请求：
- **用户 A**：问"你好"（回答很短，1 秒生成完）
- **用户 B**：问"请帮我写一篇 5000 字的论文"（回答很长，需要 30 秒）

### TorchServe / Triton 的做法：静态 Batching

```
Batch 1: [用户 A, 用户 B] → 一起处理 → 等待 30 秒 → 一起返回
```

**问题**：
1. ❌ 用户 A 虽然 1 秒就生成完了，但必须等用户 B 的 30 秒
2. ❌ 显存一直被占用，即使 A 已经完成
3. ❌ 新用户 C 到来时，必须等这个 Batch 完全结束

**类比**：就像电梯里有个人要去 30 楼，虽然你在 2 楼就该下了，但电梯不会停，你得陪他去 30 楼再一起回来。

### vLLM 的做法：Continuous Batching（连续批处理）

```
初始状态：[用户 A, 用户 B] 一起处理
1 秒后：   用户 A 完成 → 立即返回给 A → 释放显存
           [用户 B] 继续处理
1.1 秒后： 用户 C 到达 → 立即插入 → [用户 B, 用户 C] 一起处理
30 秒后：  用户 B 完成 → 返回给 B
```

**优势**：
- ✅ 短请求立即返回，延迟低
- ✅ 显存动态释放，利用率高
- ✅ 新请求随时插入，GPU 不闲置
- ✅ 吞吐量提升 **10-20 倍**

**类比**：就像滴滴拼车，乘客 A 到站就下车，司机立即接新乘客 C 上车，车永远满载运行。

### 技术实现对比

| 特性 | 静态 Batching | Continuous Batching |
|-----|--------------|---------------------|
| **Batch 大小** | 固定（如 Batch=8） | 动态变化（1~32+） |
| **请求处理** | 等待凑够 N 个请求才开始 | 请求到达立即处理 |
| **完成处理** | 等所有请求完成才返回 | 单个完成立即返回 |
| **显存占用** | 按最长请求分配 | 按实际需要动态分配 |
| **空转时间** | 高（等待凑 Batch） | 低（随到随处理） |

---

## 三、核心区别：显存管理机制

### 传统方式（TorchServe / Triton）

**预分配策略**：
```python
# 伪代码
max_seq_length = 2048
batch_size = 8
kv_cache_size = batch_size * max_seq_length * hidden_size
memory = torch.zeros(kv_cache_size)  # 一次性分配
```

**问题**：
1. ❌ 如果实际只用了 512 tokens，剩余 1536 tokens 的显存被浪费
2. ❌ 长文本和短文本混在一起时，显存碎片化严重
3. ❌ 很容易 OOM（显存溢出）

### vLLM 的 PagedAttention

**核心思想**：像操作系统管理虚拟内存一样管理 GPU 显存。

**分页机制**：
```
传统方式：[████████████████████████]  连续大块显存
         必须预分配，不能动态调整

PagedAttention：[██][  ][██][██][  ][██]  分散的页
               按需分配，用多少申请多少
```

**技术细节**：

1. **KV Cache 分页存储**
   ```python
   # 传统方式
   kv_cache = tensor([seq_len, hidden_size])  # 连续存储

   # PagedAttention
   kv_cache = [
       page_1: tensor([block_size, hidden_size]),
       page_3: tensor([block_size, hidden_size]),
       page_7: tensor([block_size, hidden_size]),
   ]  # 非连续存储，通过页表映射
   ```

2. **逻辑地址 → 物理地址映射**
   ```
   请求 A 的 token 位置:  [0, 1, 2, 3, 4, ...]
                          ↓  ↓  ↓  ↓  ↓
   实际显存页:           [P1, P1, P1, P5, P5, ...]
   ```

**优势**：
- ✅ 显存利用率接近 100%（传统方式 ~60%）
- ✅ 可以在同样的显卡上服务更多并发请求
- ✅ 长短文本混合时不会浪费显存

:::info 类比：酒店房间管理
- **传统方式**：给每个客人预定一个总统套房（2048㎡），即使他只用了 200㎡
- **PagedAttention**：客人需要多少房间就分配多少，用完立即回收给下一个客人
:::

---

## 四、性能对比

### 基准测试环境
- **模型**：Llama-2-13B
- **硬件**：NVIDIA A100 40GB
- **场景**：1000 个并发请求，平均输入 128 tokens，输出 256 tokens

### 结果对比

| 指标 | TorchServe | Triton (纯 CUDA) | vLLM |
|-----|-----------|-----------------|------|
| **吞吐量 (QPS)** | 12 | 18 | **215** |
| **平均延迟 (s)** | 8.5 | 5.2 | **1.3** |
| **P99 延迟 (s)** | 15.3 | 10.8 | **2.1** |
| **显存利用率** | 62% | 71% | **94%** |
| **最大并发数** | 32 | 48 | **256** |

:::tip 为什么差距这么大？
vLLM 通过 Continuous Batching + PagedAttention 双重优化，在 LLM 推理场景下可以达到 **10-20 倍** 的性能提升。
:::

---

## 五、适用场景总结

### 选择 vLLM 的场景 ✅

1. **搭建 ChatGPT 类的对话服务**
   - 需要高并发、低延迟
   - 请求长度差异大（有人问天气，有人要写代码）

2. **部署开源 LLM（Llama/Qwen/Mistral）**
   - 显存有限，想最大化吞吐量
   - 用户请求随时到达，不能批量处理

3. **API 服务商**
   - 需要服务成千上万的用户
   - 成本敏感（更少的 GPU 卡 = 更低成本）

### 选择 TorchServe 的场景 ✅

1. **部署多种类型的模型**
   - 同时有图像、语音、文本模型
   - 需要统一的管理界面

2. **PyTorch 生态内项目**
   - 团队熟悉 PyTorch
   - 需要快速原型验证

3. **企业内部服务**
   - 并发量不高（<100 QPS）
   - 更看重稳定性和易用性

### 选择 Triton 的场景 ✅

1. **复杂的 AI 平台**
   - 需要组合多个模型（图像+文本+语音）
   - 需要模型集成（Ensemble）功能

2. **极致性能追求**
   - 愿意花时间配置 TensorRT、FasterTransformer
   - 需要 C++ 客户端对接

3. **NVIDIA 硬件深度绑定**
   - 使用 Tensor Core、MIG 等高级特性
   - 需要官方技术支持

---

## 六、混合部署方案

### Triton + vLLM 后端

由于 vLLM 在 LLM 推理上太强，NVIDIA 已经支持将 vLLM 作为 Triton 的后端插件使用。

**架构示例**：
```
                    ┌─────────────┐
                    │   Triton    │
                    │   Inference │
                    │   Server    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
      │  YOLO   │    │  vLLM   │    │ Whisper │
      │ Backend │    │ Backend │    │ Backend │
      └────┬────┘    └────┬────┘    └────┬────┘
           │               │               │
      图像识别        文本生成         语音识别
```

**优势**：
- ✅ 对外只暴露一个 Triton 接口
- ✅ 图像/语音用 Triton 原生后端
- ✅ LLM 用 vLLM 后端获得极致性能
- ✅ 统一监控和管理

**配置示例**：
```python
# Triton 配置文件
name: "llama2"
backend: "vllm"
parameters: {
  key: "model"
  value: { string_value: "meta-llama/Llama-2-13b-hf" }
}
parameters: {
  key: "tensor_parallel_size"
  value: { string_value: "2" }
}
```

---

## 七、快速决策表

| 你的需求 | 推荐方案 | 原因 |
|---------|---------|------|
| 只部署 LLM（如 Llama/Qwen） | **vLLM** | 性能最优，开箱即用 |
| LLM + 图像/语音模型 | **Triton + vLLM 后端** | 兼顾通用性和 LLM 性能 |
| PyTorch 模型快速验证 | **TorchServe** | 最简单，生态完整 |
| 需要 TensorRT 加速 | **Triton** | NVIDIA 官方深度优化 |
| 成本敏感（GPU 资源有限） | **vLLM** | 显存利用率最高 |
| 并发量 <50 QPS | **TorchServe** | 足够用，不需要过度优化 |
| 并发量 >500 QPS | **vLLM** | 唯一能扛住的选择 |

---

## 八、技术演进趋势

### 当前状态（2024-2025）

1. **vLLM 一骑绝尘**
   - 开源社区最活跃的 LLM 推理项目
   - Meta、Google、腾讯等大厂生产环境使用

2. **Triton 拥抱 vLLM**
   - 官方支持 vLLM 后端
   - 弥补了 LLM 推理的短板

3. **TorchServe 稳健发展**
   - 依然是 PyTorch 模型部署的首选
   - 但在 LLM 场景逐渐被边缘化

### 未来方向

- **vLLM**：可能扩展到多模态 LLM（LLaVA、Qwen-VL）
- **Triton**：继续深化异构硬件支持（AMD、Intel）
- **TorchServe**：可能也会集成类似 Continuous Batching 的特性

---

## 九、总结

### 一句话总结

> **如果你只跑大语言模型（Llama/Qwen/Mistral），闭眼选 vLLM。**
> **如果你要搭建包含图像、语音、文本的全能 AI 平台，选 Triton（内部可以挂 vLLM）。**
> **如果你只是想快速验证一个 PyTorch 模型，选 TorchServe。**

### 核心记忆点

1. **vLLM = LLM 专科医院**（只看心脏病，但世界第一）
2. **Continuous Batching** 是 vLLM 的杀手锏（电梯 vs 拼车）
3. **PagedAttention** 让显存利用率从 60% → 94%
4. **性能差距**：vLLM 可以比通用框架快 10-20 倍

---

## 参考资料

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [TorchServe 文档](https://pytorch.org/serve/)
- [NVIDIA Triton 文档](https://github.com/triton-inference-server/server)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM 论文
