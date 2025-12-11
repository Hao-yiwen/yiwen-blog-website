---
title: PPO 近端策略优化训练指南
sidebar_position: 26
tags: [PPO, RLHF, 强化学习, LLM对齐, 微调, DPO]
---

# PPO (Proximal Policy Optimization)

> **摘要**：PPO（Proximal Policy Optimization，近端策略优化）是 OpenAI 于 2017 年提出的强化学习算法。如果说 DPO 是现在的"当红炸子鸡"，那 PPO 就是打造了 ChatGPT 帝国的"开国元勋"。直到今天，尽管 DPO 流行，但如果你想训练一个逻辑推理能力极强（如 OpenAI o1, DeepSeek-R1）的模型，PPO 依然是绕不过去的高山。

本文将从三个层面来讲解 PPO：**它的核心直觉（通俗版）**、**它的数学魔法（进阶版）**，以及**它在 LLM 里的实际操作（实战版）**。

-----

## 1. 核心直觉：为什么要发明 PPO？

在 PPO 出现之前，强化学习（RL）主要面临一个巨大的痛点：**步子迈多大合适？**

想象你在蒙着眼睛走钢丝（训练模型）：

* **步子太小（学习率低）：** 你走得太慢，训练几万年都不收敛。
* **步子太大（学习率高）：** 你试图根据上一步的经验迈一大步，结果直接掉下悬崖（模型参数崩坏，输出乱码），而且再也爬不上来了。

**TRPO (PPO 的前身)** 试图解决这个问题，它用复杂的数学约束（库尔贝克-莱布勒散度，KL Divergence）划定一个"安全区域"，告诉你"每一步只能在圈子里动"。效果很好，但**计算量大到令人发指**。

**PPO 的出现：**

PPO 继承了"安全区域"的思想，但它用了一个极简的工程技巧（**Clipping / 裁剪**）替代了复杂的数学计算。

**PPO 的潜台词是：**

> "你可以改进，但不要改得离刚才的自己太远。如果你这步改得太猛了，我就把多出来的部分'剪掉'不算，强行让你慢下来。"

-----

## 2. 核心机制：PPO 到底做了什么？（进阶版）

PPO 的核心是一个特殊的**损失函数（Loss Function）**。它的魔法在于控制**新策略（New Policy）**和**旧策略（Old Policy）**之间的差异。

### 2.1 重要的比率（Ratio）

PPO 会计算一个比率：

$$
r_t(\theta) = \frac{\text{新策略生成该动作的概率}}{\text{旧策略生成该动作的概率}}
$$

* 如果 $r = 1$，说明没变化。
* 如果 $r > 1$，说明新策略觉得这个动作比旧策略更好（更倾向于做这个动作）。

### 2.2 剪裁（Clipping）—— PPO 的灵魂

在更新模型参数时，PPO 会看这个 $r$ 值。它设定了一个范围，通常是 $[1-\epsilon, 1+\epsilon]$（比如 $\epsilon$ 是 0.2，范围就是 0.8 到 1.2）。

* **情况 A：** 模型做了一个动作，得到了**正向奖励**（做得好）。
    * 模型想大幅提升这个动作的概率（比如 $r$ 变成了 1.5）。
    * **PPO 说：** "慢着！上限就是 1.2。超过 1.2 的部分我不认，哪怕你这一步走得再好，我也只按 1.2 的幅度给你奖励。"
    * **目的：** 防止模型因为一次偶然的成功就过度自信，彻底改变策略。

* **情况 B：** 模型做了一个动作，得到了**负向奖励**（做错了）。
    * 模型想大幅降低这个动作的概率（比如 $r$ 变成了 0.5）。
    * **PPO 说：** "慢着！下限是 0.8。别因为一次失败就吓得彻底不敢动了。"
    * **目的：** 防止模型因为一次失败就彻底废掉某个能力。

通过这种简单粗暴的**"截断"**，PPO 保证了模型的每一次更新都在一个"可控、平稳"的范围内（Proximal / 近端）。

### 2.3 PPO 的目标函数

PPO 的 Clipped 目标函数：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数估计（Advantage Estimation）
- $\epsilon$ 是裁剪参数（通常 0.1~0.2）

-----

## 3. 实战流程：PPO 在 LLM 训练中是怎么跑的？

这是大家最关心的部分：**ChatGPT 是怎么用 PPO 练出来的？**

这个过程被称为 **RLHF（Reinforcement Learning from Human Feedback）**。在 PPO 阶段，你的显卡里通常需要同时加载 **4 个模型**（这也是为什么 PPO 显存消耗巨大的原因）。

### 3.1 舞台上的四个角色

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF-PPO 四模型架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │    Actor     │    │   Critic     │                          │
│  │  (策略模型)   │    │  (价值模型)   │                          │
│  │   可训练 ✓   │    │   可训练 ✓   │                          │
│  └──────────────┘    └──────────────┘                          │
│         │                   │                                   │
│         ▼                   ▼                                   │
│     生成回答              估计奖励                                │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │   Reward     │    │  Reference   │                          │
│  │  (奖励模型)   │    │  (参考模型)   │                          │
│  │   冻结 ✗     │    │   冻结 ✗     │                          │
│  └──────────────┘    └──────────────┘                          │
│         │                   │                                   │
│         ▼                   ▼                                   │
│     打分评价              KL 惩罚                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

1. **Actor (演员/策略模型)：** 我们要训练的那个 LLM（比如 Llama-3-8B）。
2. **Ref (参考模型)：** 原始的 LLM，完全冻结参数不动。用来做对比，防止 Actor 练傻了。
3. **Critic (评论家/价值模型)：** 预测当前状态未来能得多少分。
4. **Reward (奖励模型/裁判)：** 替人类打分，告诉 Actor 这句话说得好不好。

### 3.2 训练循环（Step-by-Step）

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO 训练循环流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: Rollout (生成)                                   │  │
│  │ ┌────────┐      ┌────────┐      ┌────────┐              │  │
│  │ │ Prompt │ ───▶ │ Actor  │ ───▶ │Response│              │  │
│  │ └────────┘      └────────┘      └────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: Evaluation (打分)                                │  │
│  │ ┌────────┐                    ┌────────────────────────┐ │  │
│  │ │ Reward │ ──▶ Score ──────▶ │ Final = Score - KL惩罚 │ │  │
│  │ └────────┘                    └────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 3: Advantage Estimation (计算优势)                  │  │
│  │ Critic 对比 "实际得分" vs "预期得分"                      │  │
│  │ Advantage = 实际 - 预期 (正值=超常发挥)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 4: Optimization (更新)                              │  │
│  │ 使用 PPO Clipping 更新 Actor 和 Critic                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 1: 生成 (Rollout)

* 给 Actor 一个问题（Prompt）。
* Actor 生成一个回答。
* Ref 模型也看一眼这个回答，计算一个概率。

#### Step 2: 打分 (Evaluation)

* **Reward 模型**给回答打一个分（比如 +10 分）。
* **KL 惩罚 (关键)：** PPO 会计算 Actor 生成的字和 Ref 生成的字差别大不大。如果差别太大（Actor 开始胡言乱语），就扣分。
* 最终得分 = Reward分数 - KL惩罚。

#### Step 3: 计算优势 (Advantage Estimation)

* Critic 模型登场。它会对比"实际得分"和"它预期的得分"。
* 如果实际得分 > 预期得分，说明这是一个"超常发挥"的好动作，Advantage 是正的。

#### Step 4: 更新 (Optimization via PPO)

* 利用计算出的 Advantage，使用 PPO 的**剪裁机制**来更新 Actor 的参数。
* 同时更新 Critic 的参数，让它下次预测得更准。

-----

## 4. 核心代码实现 (PyTorch)

以下是 PPO Loss 的核心 PyTorch 实现：

```python
import torch
import torch.nn.functional as F

def ppo_loss(
    old_log_probs,      # 旧策略的 log 概率
    new_log_probs,      # 新策略的 log 概率
    advantages,         # 优势函数
    clip_epsilon=0.2    # 裁剪参数
):
    """
    PPO 的核心 Loss 计算

    参数说明:
    old_log_probs: 旧策略（rollout时）的 log 概率
    new_log_probs: 新策略（当前参数）的 log 概率
    advantages: 优势函数值，正值表示好动作，负值表示坏动作
    clip_epsilon: 裁剪范围，通常 0.1~0.2
    """
    # 1. 计算概率比 r(θ) = π_new / π_old
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 2. 未裁剪的目标 (Surrogate 1)
    surr1 = ratio * advantages

    # 3. 裁剪后的目标 (Surrogate 2)
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    # 4. PPO Loss：取两者最小值（保守策略）
    # 为什么取 min？
    # - 如果 advantage > 0（好动作），我们希望 ratio 增大，但不要超过 1+ε
    # - 如果 advantage < 0（坏动作），我们希望 ratio 减小，但不要低于 1-ε
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_kl_penalty(log_probs_policy, log_probs_ref, kl_coef=0.1):
    """
    计算 KL 散度惩罚，防止模型偏离参考模型太远

    KL(π_ref || π_policy) ≈ Σ π_ref * (log π_ref - log π_policy)
    """
    kl_div = torch.exp(log_probs_ref) * (log_probs_ref - log_probs_policy)
    return kl_coef * kl_div.sum(dim=-1).mean()


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    使用 GAE (Generalized Advantage Estimation) 计算优势函数

    参数说明:
    rewards: 每个时间步的奖励
    values: Critic 模型预测的价值
    gamma: 折扣因子
    lam: GAE 的 λ 参数
    """
    advantages = []
    gae = 0

    # 从后往前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD 误差
        delta = rewards[t] + gamma * next_value - values[t]
        # GAE
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)


# ==========================================
# 完整的 PPO 训练步骤示例
# ==========================================
def ppo_train_step(
    actor_model,
    critic_model,
    ref_model,
    reward_model,
    prompts,
    optimizer_actor,
    optimizer_critic,
    clip_epsilon=0.2,
    kl_coef=0.1
):
    """
    PPO 的一个完整训练步骤
    """
    # Step 1: Rollout - 使用当前 Actor 生成回答
    with torch.no_grad():
        responses, old_log_probs = actor_model.generate_with_log_probs(prompts)
        ref_log_probs = ref_model.get_log_probs(prompts, responses)

    # Step 2: 计算奖励
    with torch.no_grad():
        rewards = reward_model.score(prompts, responses)
        # 添加 KL 惩罚
        kl_penalty = kl_coef * (old_log_probs - ref_log_probs)
        final_rewards = rewards - kl_penalty

    # Step 3: 计算优势函数
    with torch.no_grad():
        values = critic_model(prompts, responses)
        advantages = compute_advantages(final_rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Step 4: PPO 更新
    # 4.1 更新 Actor
    new_log_probs = actor_model.get_log_probs(prompts, responses)
    actor_loss = ppo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon)

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # 4.2 更新 Critic
    new_values = critic_model(prompts, responses)
    critic_loss = F.mse_loss(new_values, final_rewards)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_kl": kl_penalty.mean().item()
    }
```

-----

## 5. PPO 的优缺点总结

为什么现在 DPO 火了，但 DeepSeek 和 OpenAI 还是离不开 PPO？

### 5.1 PPO 的优点（为什么它是皇冠上的明珠）

| 优点 | 说明 |
|:-----|:-----|
| **上限极高（探索能力）** | PPO 是**在线（Online）**算法。模型是在实时生成的过程中学习的。它有机会尝试出数据集中不存在的解法 |
| **防止崩坏** | 由于有 Trust Region（信任域/裁剪），它极其稳定，不容易在训练后期发生模型能力骤降（Catastrophic Forgetting） |
| **适合复杂推理** | 对于数学、代码等需要多步推理的任务，PPO 能让模型自我探索最优路径 |

**探索能力举例：**

> 教模型做数学题。DPO 只能学你给它的解题步骤；PPO 可以在尝试中发现一种你没教过但更简便的解法，只要 Reward Model 判定答案正确即可。

### 5.2 PPO 的缺点（为什么大家想淘汰它）

| 缺点 | 说明 |
|:-----|:-----|
| **工程噩梦** | 这一套流程涉及到 4 个模型的协同，数据流极其复杂。代码实现非常难，很容易写出 Bug 但不报错，只是效果不好 |
| **超参敏感** | 学习率、KL 系数、Clip 范围……有几十个参数要调。调错一个，模型就废了 |
| **慢且贵** | 相比 DPO，PPO 的采样效率低，训练时间长，显存占用大 |
| **显存爆炸** | 4 个模型同时加载，7B 模型需要 80GB+ 显存 |

-----

## 6. PPO vs DPO 对比

| 特性 | PPO | DPO |
|:-----|:----|:----|
| **训练方式** | 在线（Online） | 离线（Offline） |
| **模型数量** | 4 个（Actor, Critic, Reward, Ref） | 2 个（Policy, Ref） |
| **是否需要生成** | 是（Rollout/Sampling） | 否（直接用现成数据） |
| **训练速度** | 慢 | 快（通常是 PPO 的几倍） |
| **显存占用** | 极大 | 较小 |
| **超参敏感度** | 极高 | 较低 |
| **探索能力** | 强（可发现新解法） | 弱（只能学数据中的解法） |
| **适用场景** | 复杂推理（数学、代码） | 偏好对齐（聊天、安全） |

-----

## 7. 实战建议

### 7.1 什么时候用 PPO？

* 你想训练模型进行**复杂的逻辑推理**（数学、代码、科研）
* 需要模型**自我探索**出最优路径（如 OpenAI o1/o3, DeepSeek R1）
* 你有充足的算力资源

### 7.2 什么时候用 DPO？

* 你只是想让模型**说话好听、符合人类偏好**（写文案、聊天）
* 算力有限
* 想要快速迭代

### 7.3 超参数建议

| 参数 | 推荐值 | 说明 |
|:-----|:-------|:-----|
| `clip_epsilon` | 0.1 ~ 0.2 | 太大会不稳定，太小会收敛慢 |
| `kl_coef` | 0.01 ~ 0.1 | 控制 KL 惩罚强度 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` (GAE) | 0.95 | GAE 的 λ 参数 |
| `learning_rate` | 1e-6 ~ 5e-5 | Actor 学习率，通常比 SFT 更小 |
| `batch_size` | 尽量大 | 提高训练稳定性 |

-----

## 8. 总结

**PPO 就像一位严谨的"太极大师"。** 它讲究"以慢打快"，每一步都走得很稳（Clipping），不求一步登天，但求步步为营。

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 训练方法选择指南                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      你的需求是什么？                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                      │
│          ┌───────────────┴───────────────┐                     │
│          ▼                               ▼                      │
│  ┌──────────────┐               ┌──────────────┐               │
│  │  偏好对齐     │               │  复杂推理     │               │
│  │  聊天/安全    │               │  数学/代码    │               │
│  └──────────────┘               └──────────────┘               │
│          │                               │                      │
│          ▼                               ▼                      │
│  ┌──────────────┐               ┌──────────────┐               │
│  │     DPO      │               │     PPO      │               │
│  │   快速简单    │               │  上限更高     │               │
│  └──────────────┘               └──────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

* 如果你只是想让模型**说话好听、符合人类偏好**（写文案、聊天），**DPO** 这种"速成班"已经足够好了，甚至更好。
* 如果你想训练模型进行**复杂的逻辑推理**（数学、代码、科研），需要模型自我探索出最优路径（如 OpenAI o1/o3, DeepSeek R1），那么 **PPO（及其变体 GRPO）** 依然是目前唯一通向真理的道路。

在 2024 年及以后，**SFT + DPO/PPO** 几乎是定制大模型的标准流程。SFT 奠定基础，对齐方法（DPO 或 PPO）锦上添花。
