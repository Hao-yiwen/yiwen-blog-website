---
title: GRPO 组相对策略优化详解
sidebar_position: 26
tags: [GRPO, PPO, RLHF, 强化学习, DeepSeek, LLM对齐, 推理模型]
---

# GRPO (Group Relative Policy Optimization) 详解

> **摘要**：GRPO（组相对策略优化）是一种用于大语言模型强化学习的高效算法。该算法由 DeepSeek 团队在 DeepSeekMath 论文中提出，并在 DeepSeek-V3 和 DeepSeek-R1 等高性能模型的训练中发挥了核心作用。GRPO 的核心突破在于：它**摒弃了传统 PPO 算法中必须的"评论家"（Critic）模型**，通过从"一组"生成的输出中计算相对优势，显著降低了训练时的显存占用和计算成本。

-----

## 1. 背景与动机

在传统的 RLHF（Reinforcement Learning from Human Feedback）流程中，**PPO** 是主流算法。然而，PPO 在训练超大参数量的 LLM 时面临巨大的资源瓶颈。

### 1.1 PPO 的痛点

标准的 PPO 训练通常需要维护四个模型：

1. **Actor (策略模型):** 正在训练的模型。
2. **Reference (参考模型):** 用于计算 KL 散度，防止模型跑偏。
3. **Reward (奖励模型):** 用于打分。
4. **Critic (价值模型/评论家):** 用于估计当前状态的价值 $V(s)$，以计算优势函数（Advantage）。

**问题在于：** Critic 模型通常与 Actor 模型大小相当。如果 Actor 是一个 70B 的模型，Critic 也需要是 70B。这意味着训练时的显存需求几乎翻倍，且增加了计算和通信开销。

### 1.2 GRPO 的解决方案

GRPO 提出：**我们真的需要一个单独的神经网络来估计价值吗？**

答案是：不需要。我们可以通过对同一个 Prompt 采样**一组（Group）**输出，用这组输出的奖励均值作为基线（Baseline），从而计算优势。

-----

## 2. GRPO 核心机制

GRPO 的工作流程可以概括为：**分组采样 -> 奖励计算 -> 组内标准化 -> 策略更新**。

### 2.1 算法流程

对于每一个输入问题（Prompt）$q$，GRPO 执行以下步骤：

#### 步骤一：分组采样 (Group Sampling)

从当前策略 $\pi_\theta$ 中，针对同一个问题 $q$，采样生成 $G$ 个不同的输出 $\{o_1, o_2, ..., o_G\}$。通常 $G$ 取值为 64 或更高。

#### 步骤二：奖励评估 (Reward Evaluation)

使用奖励模型（Reward Model）或基于规则的验证器（Rule-based Verifier，如数学题答案检查），对这 $G$ 个输出进行打分，得到奖励集合 $\{r_1, r_2, ..., r_G\}$。

*注意：此时通常也会计算 KL 散度惩罚，加到奖励中。*

#### 步骤三：优势估计 (Advantage Estimation)

这是 GRPO 的核心。它不使用 Critic 模型预测的 $V(s)$，而是利用组内数据的统计特征来计算优势 $A_i$。

对于第 $i$ 个输出，其优势计算公式如下：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}
$$

其中：
- $\text{mean}(\dots)$ 是这组奖励的平均值。
- $\text{std}(\dots)$ 是这组奖励的标准差。

**直观理解：** 如果某个输出的奖励高于这组输出的平均水平，它的优势就是正的（鼓励该行为）；反之则是负的（抑制该行为）。

#### 步骤四：策略优化 (Policy Optimization)

使用计算出的优势 $A_i$，配合 PPO 中的裁剪（Clipping）机制来更新模型参数。目标函数（Loss Function）如下：

$$
J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \mathbb{D}_{KL} \right) \right]
$$

-----

## 3. GRPO 与 PPO 的详细对比

| 特性 | PPO | GRPO |
| :--- | :--- | :--- |
| **模型数量** | **4个** (Actor, Ref, Reward, Critic) | **3个** (Actor, Ref, Reward) |
| **显存占用** | 极高 (Critic 模型占用大量显存) | **低** (省去了一个与 Actor 同等规模的模型) |
| **基线 (Baseline)** | 依赖神经网络 Critic 估计 $V(s)$ | 依赖 Group 内输出的**平均奖励** |
| **计算复杂度** | 需训练 Critic，需前向/反向传播 | 仅需推理生成 $G$ 个样本，无需训练 Critic |
| **适用场景** | 通用 RLHF | 特别适合推理、数学、代码等**结果可验证**的任务 |

:::tip
GRPO 的 Reward 模型甚至可以是不可导的规则脚本，例如数学题的答案验证器。
:::

-----

## 4. 为什么 GRPO 对推理模型至关重要？

DeepSeek-R1 等强推理模型的成功很大程度上归功于 GRPO，原因如下：

### 4.1 探索能力 (Exploration)

通过对每个 Prompt 生成一组（例如 64 个）输出，模型被迫探索不同的解题路径。

### 4.2 自洽性验证

在数学或代码任务中，答案通常是客观的。如果一组输出中有 10 个是对的，54 个是错的，GRPO 会极大地增强那 10 个正确路径的权重。

### 4.3 Outcome Reward (结果奖励)

传统的 PPO 往往依赖稠密的 Process Reward（过程奖励，每一步都打分），这很难标注。GRPO 允许只使用 Outcome Reward（只看最终答案对不对），通过组内对比，模型能自动学会哪些步骤导致了正确的结果。

:::info 关键洞察
GRPO 将强化学习从"预测价值"转变为"比较优劣"。在解题场景下，"比别的尝试做得好"往往比"达到某个绝对分数"更易于学习。
:::

-----

## 5. 代码实现伪代码

为了帮助理解，以下是 GRPO 核心逻辑的 Python 伪代码：

```python
def grpo_step(prompts, actor_model, reward_function):
    # 1. 分组采样 (Group Sampling)
    # 针对每个 prompt 生成 G 个 outputs
    # inputs shape: [batch_size, G, seq_len]
    outputs = actor_model.generate(prompts, num_return_sequences=G)

    # 2. 计算奖励 (Reward Calculation)
    # rewards shape: [batch_size, G]
    rewards = reward_function(prompts, outputs)

    # 3. 计算优势 (Advantage Computation)
    # 在组维度 (Group dimension) 上进行归一化
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)

    # 避免除以零
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)

    # 4. 计算 Loss 并更新
    # 这里的 Loss 包含 Policy Loss 和 KL 散度
    loss = compute_grpo_loss(actor_model, prompts, outputs, advantages)

    loss.backward()
    optimizer.step()
```

-----

## 6. 总结

GRPO 是一种**去 Critic 化**的高效强化学习算法。它通过"以群为鉴"（Group Relative）的方式，解决了大模型 RL 训练中显存开销大、训练不稳定的问题。

**核心价值：**

1. **省钱省力：** 移除 Critic 模型，大幅降低显存和计算资源需求。
2. **简单有效：** 算法逻辑比 PPO 更简洁，且在推理类任务上表现卓越。
3. **支持长思维链：** 非常适合训练模型产生 Long Chain-of-Thought (CoT)，因为它可以基于最终结果有效地反向传播优势信号。

-----

## 参考资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
