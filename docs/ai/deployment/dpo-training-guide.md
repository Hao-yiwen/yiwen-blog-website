---
title: DPO 直接偏好优化训练指南
sidebar_position: 25
tags: [DPO, RLHF, PPO, 偏好优化, LLM对齐, 微调]
---

# DPO

> **摘要**：在 LLM 的训练流程中，SFT 教会了模型"说话"，而 DPO（Direct Preference Optimization）则教会了模型"如何得体地说话"。本文将深入浅出地拆解 DPO 的核心原理、数学本质以及它为何能在 2023 年横空出世后迅速取代复杂的 RLHF (PPO)。

-----

## 1. 背景：为什么 SFT 还不够？

如果你训练过大模型，你一定熟悉 **Pre-train（预训练）** 和 **SFT（监督微调）**。
SFT 的本质是"模仿"。我们给模型成千上万个 `(User, Assistant)` 的问答对，让模型预测下一个 token。

但是，SFT 有一个致命弱点：**它不知道什么是"更好"**。

  * **回答 A**：准确但啰嗦，像个老学究。
  * **回答 B**：准确且简洁，像个专家。

在 SFT 模型眼里，只要能接上下一个字，A 和 B 没啥区别。但对于人类来说，我们显然偏好 B。为了让模型符合人类的价值观（有用性、安全性、简洁性），我们需要 **Human Alignment（人类对齐）**。

在 DPO 出现之前，这个领域的霸主是 **RLHF (PPO)**。

### 1.1 PPO 的痛点

OpenAI 使用 PPO（Proximal Policy Optimization）训练出了 ChatGPT。虽然效果好，但 PPO 的训练过程极其痛苦：

  * **太复杂**：需要同时维护 4 个模型（Actor, Critic, Reward, Reference）。
  * **不稳定**：超参数极其敏感，训练经常不收敛。
  * **极慢**：训练过程中模型需要实时生成（Rollout/Sampling）文本，这不仅慢，还极其吃显存。

直到 2023 年 5 月，斯坦福大学提出了 **DPO (Direct Preference Optimization)**，一切都变了。

-----

## 2. DPO 的核心思想：大道至简

DPO 的论文标题非常有意思：《*Your Language Model is Secretly a Reward Model*》（你的语言模型其实私底下是个奖励模型）。

它的核心逻辑是：**我们不需要显式地训练一个 Reward Model（奖励模型）。我们可以直接利用"人类偏好数据"来优化语言模型本身。**

### 2.1 数据的变化

DPO 不再是"填空题"，而是"选择题"。我们需要的数据格式如下：

```json
{
  "prompt": "如何制作危险爆炸物？",
  "chosen": "我无法提供该信息，这违反了安全准则...",  // 好的回答 (y_w)
  "rejected": "首先你需要去化工店购买..."             // 坏的回答 (y_l)
}
```

DPO 的目标很简单：**提高 `chosen` 回答的生成概率，同时降低 `rejected` 回答的生成概率。**

-----

## 3. 数学原理：这公式到底在算什么？

DPO 的损失函数看起来很吓人，但拆解开来非常优雅：

$$
L_{DPO} = - \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

别慌，我们把它翻译成伪代码逻辑，它其实就是：

$$
\text{Loss} = -\log \text{Sigmoid} \Big( \beta \times [ (\text{Policy}_{diff}) - (\text{Ref}_{diff}) ] \Big)
$$

这里的核心博弈是"左右互搏"：

1.  **$\pi_\theta$ (Policy Model)**：我们要训练的学生模型。
2.  **$\pi_{ref}$ (Reference Model)**：冻结参数的老师模型（通常是 SFT 后的底模）。

**公式在做什么？**
它计算了两个"差距"：

  * **差距 A**：学生模型觉得"好答案"比"坏答案"好多少？（$\log P_{policy}(y_w) - \log P_{policy}(y_l)$）
  * **差距 B**：老师模型觉得"好答案"比"坏答案"好多少？（$\log P_{ref}(y_w) - \log P_{ref}(y_l)$）

**DPO 的目的**：让差距 A **大于** 差距 B。
也就是说，**现在的模型要比原来的模型，更能分辨出什么是好、什么是坏。**

-----

## 4. 为什么 DPO 比 PPO 快？（关键点）

这是 DPO 最具革命性的地方。

  * **PPO 是"写作文"**：在训练的每一步，PPO 都要让模型**现场生成**一段文本（Sampling），然后打分，再反向传播。生成文本是一个串行的过程，极其耗时。
  * **DPO 是"阅读理解"**：数据（好回答/坏回答）是现成的。DPO 只需要把数据喂进去，进行一次前向传播（Forward），算出概率即可。**它完全不需要模型进行生成（Avoid Sampling）。**

因此，DPO 的训练速度通常是 PPO 的几倍，且显存占用大幅降低（只需要加载 2 个模型）。

-----

## 5. 核心代码实现 (PyTorch)

光说不练假把式。这是 DPO Loss 的核心 PyTorch 实现，你可以直接用到你的项目中：

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """
    参数说明:
    policy_chosen_logps: 当前模型对"好回答"的 log 概率
    policy_rejected_logps: 当前模型对"坏回答"的 log 概率
    ref_chosen_logps:     参考模型对"好回答"的 log 概率
    ref_rejected_logps:   参考模型对"坏回答"的 log 概率
    beta:                 超参数，控制偏离参考模型的程度 (通常 0.1)
    """

    # 1. 计算当前模型的偏好程度 (好 - 坏)
    pi_logratios = policy_chosen_logps - policy_rejected_logps

    # 2. 计算参考模型的偏好程度 (好 - 坏)
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # 3. DPO 核心逻辑：我们希望 pi 的 logratios 比 ref 的大
    logits = pi_logratios - ref_logratios

    # 4. 计算 Loss (-log sigmoid)
    # 使用 logsigmoid 保证数值稳定性
    losses = -F.logsigmoid(beta * logits)

    # 5. 可选：计算奖励 (用于观察训练进度)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards
```

**代码逻辑图解：**
模型不需要自己判断哪个是 `chosen`，哪个是 `rejected`。这是你在 DataLoader 阶段就硬编码告诉模型的。模型只是负责计算概率，然后 Loss 函数负责"奖优罚劣"。

-----

## 6. 实战避坑指南

如果你准备上手 DPO，这里有几个经验之谈：

1.  **数据质量大于数量**：DPO 对噪声非常敏感。如果你的 `chosen` 其实写得不咋地，或者 `rejected` 其实也没那么差，模型会非常困惑。宁可要 1000 条高质量数据，不要 10 万条垃圾数据。
2.  **$\beta$ (Beta) 的选择**：通常设为 `0.1`。
      * $\beta$ 越大，模型越急于讨好人类偏好，越容易忽略语言流畅性。
      * $\beta$ 越小，模型越保守，越接近原始 SFT 模型。
3.  **显存溢出 (OOM)**：虽然比 PPO 省，但 DPO 还是要加载两个模型（Policy + Ref）。对于 7B 以上的模型，建议使用 **LoRA** 甚至 **Q-LoRA**。Ref Model 可以量化加载，因为它不需要更新参数。
4.  **Batch Size**：尽量大。因为是对比学习，Batch Size 太小会导致梯度方向不稳定。

-----

## 7. 总结

DPO 是大模型微调领域的一次"奥卡姆剃刀"式的胜利。它切掉了 PPO 中所有复杂的、不必要的部分，回归到了最本质的优化目标。

  * **SFT**：学会了知识（填鸭式教学）。
  * **DPO**：学会了品味（鉴赏式教学）。

在 2024 年及以后，如果你想定制一个符合特定价值观、说话好听的垂直领域大模型，**SFT + DPO** 几乎是标准的黄金组合。
