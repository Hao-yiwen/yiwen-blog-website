---
title: PagedAttention：vLLM 的显存管理革命
sidebar_label: PagedAttention
date: 2025-12-06
last_update:
  date: 2025-12-06
tags: [transformer, vllm, paged-attention, kv-cache, inference, memory-management]
---

# PagedAttention：vLLM 的显存管理革命

**PagedAttention** 是加州大学伯克利分校（UC Berkeley）团队在 **vLLM** 项目中提出的核心技术，它彻底改变了 LLM 推理的显存管理方式。

**一句话概括：PagedAttention 就是把操作系统的"虚拟内存分页"技术，搬到了大模型的 KV Cache 管理上。**

---

## 1. 痛点：传统 KV Cache 的"显存豪宅浪费"

在 PagedAttention 出现之前（比如 HuggingFace Transformers 的默认实现），显存管理是非常**粗放**的。

假设模型的 `max_len` 是 2048。当你发来一个请求："你好"。

* **传统做法：** 系统为了防止你后面废话连篇，必须为你预留 **2048** 个 Token 的连续显存空间。
* **实际情况：** 你只说了"你好"（2个 Token），然后模型回了"早"（1个 Token）。一共用了 3 个坑位。
* **结果：** `2048 - 3 = 2045` 个显存坑位被**锁定**了，虽然是空的，但别人不能用。

这造成了两个巨大的问题：

1. **内部碎片 (Internal Fragmentation)：** 预留了没用完的空间。
2. **外部碎片 (External Fragmentation)：** 即使显存总量够，但因为不够"连续"，无法塞进新的请求。

**据统计，传统方式下，KV Cache 的显存浪费率高达 60% - 80%。** 这意味着你的昂贵 A100 显卡，大部分显存都在"占着茅坑不拉屎"。

---

## 2. 救星：PagedAttention 的设计原理

PagedAttention 的灵感直接来源于操作系统（OS）管理内存的方式。

### 2.1 核心概念：Block（块）

它不再申请连续的巨大空间，而是把 KV Cache 切成一个个小的 **Block**。

* 例如：设定 `block_size = 16`。
* 每个 Block 可以存 16 个 Token 的 KV 数据。

### 2.2 逻辑空间 vs 物理空间

它引入了两个视角：

* **逻辑块 (Logical Blocks)：** 在用户的视角里，这句话"今天天气不错..."是连续的。
* **物理块 (Physical Blocks)：** 在 GPU 显存里，这些数据是**打散**存储的。

### 2.3 核心组件：Block Table (页表)

这就像操作系统的页表。它记录了"逻辑"到"物理"的映射关系。

**举个例子：**

假设 Prompt 是："A B C ... (共30个词)"。`block_size = 16`。

1. **前 16 个词 (0-15):** 填满 **逻辑块 0**。系统分配 **物理块 7** 给它。
2. **后 14 个词 (16-29):** 填入 **逻辑块 1**。系统分配 **物理块 3** 给它。
3. **生成新词 (30):** 填入 **逻辑块 1** 的剩余空位。
4. **生成新词 (31):** **逻辑块 1** 满了！系统申请一个新的 **物理块 9**，作为 **逻辑块 2**。

```
Block Table 示例：
┌─────────────┬─────────────┐
│ Logical Block │ Physical Block │
├─────────────┼─────────────┤
│      0      │      7      │
│      1      │      3      │
│      2      │      9      │
└─────────────┴─────────────┘
```

---

## 3. PagedAttention 到底强在哪？

有了这个机制，推理过程发生了翻天覆地的变化。

### 3.1 零浪费 = 高吞吐 (High Throughput)

* **按需分配：** 你生成多少，我给多少。绝对不预留 2048 这种傻事。
* **非连续存储：** 只要显存的角角落落里还有空的 Block，我就能塞进新的 Token。
* **结果：** 同样一张 40GB 的显卡，以前只能同时服务 10 个人（Batch Size=10），现在因为节省了 60% 的显存，可以同时服务 30 个人（Batch Size=30）。

### 3.2 内存共享 (Memory Sharing) —— 真正的杀手锏

这是 PagedAttention 最"骚"的操作。在高级采样场景（如 **Parallel Sampling** 或 **Beam Search**）中，它能省下巨量显存。

**场景：** 你让模型"写三个不同的故事开头"，Prompt 是一样的："很久很久以前"。

* **传统做法：** 把"很久很久以前"的 KV Cache 复制 3 份，分别给 3 个请求。
* **PagedAttention 做法：**
    * **Prompt 阶段：** 3 个请求的 Block Table，都指向**同一个**物理块（存着"很久很久以前"）。（引用计数 = 3）
    * **生成阶段 (Copy-on-Write)：**
        * 请求 A 生成了"有一座山"。它申请自己的新 Block。
        * 请求 B 生成了"有一个人"。它申请自己的新 Block。
    * **结果：** 公共前缀（Prompt）的显存只有一份！不需要复制。

```
Memory Sharing 示意图：

Request A ──┐
Request B ──┼──► Physical Block 5 (共享 Prompt: "很久很久以前")
Request C ──┘
              │
              ├──► Physical Block 8 (Request A: "有一座山")
              ├──► Physical Block 2 (Request B: "有一个人")
              └──► Physical Block 6 (Request C: "有一条龙")
```

---

## 4. 它是如何计算的？（算子层面的挑战）

你可能会问：*"虽然省显存了，但 Attention 计算需要读取矩阵啊，现在数据都散落在物理内存的各个角落，怎么算？"*

这就是 vLLM 团队写的 **Custom CUDA Kernel**（定制 CUDA 算子）厉害的地方。

在 Decode 阶段，这个算子是这样工作的：

1. **输入：** 当前 Token 的 Query ($Q$)。
2. **查表：** 拿到当前请求的 **Block Table**（比如 `[7, 3, 9]`）。
3. **抓取 (Gather)：** 算子去显存地址 7 把前 16 个 KV 抓来，去地址 3 把中间 16 个 KV 抓来，去地址 9 把最后几个 KV 抓来。
4. **计算：** 在 GPU 的计算核心（SRAM/Registers）里进行 Attention 计算。
5. **输出：** 得到结果。

虽然"到处抓数据"比"直接读取连续数据"稍微麻烦一点点，但因为 decode 阶段主要瓶颈是带宽（Bandwidth-bound）而不是计算延迟，而且显存利用率的大幅提升带来的吞吐量收益，远远盖过了这点微小的开销。

---

## 5. 总结

| 问题 | 答案 |
| :--- | :--- |
| **PagedAttention 是什么？** | 不把 KV Cache 当作一个"大张量"，而是当作"一堆小方块"来管理的算法 |
| **解决显存碎片化** | 像玩俄罗斯方块一样填满显存，不再有空隙 |
| **解决并发瓶颈** | 节省出的显存可以用来塞入更大的 Batch Size |
| **共享前缀** | 多个请求可以共用同一份物理 KV Cache（如多轮对话、Beam Search） |

**这就是为什么 vLLM 能比 HuggingFace 标准代码快 2-4 倍的核心原因。它不是算得更快，而是它能在同一辆车上塞进更多的乘客。**
