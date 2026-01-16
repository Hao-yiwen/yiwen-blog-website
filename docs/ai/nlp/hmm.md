---
title: 隐马尔可夫模型 (HMM) 详解
sidebar_label: 隐马尔可夫模型
sidebar_position: 3
date: 2025-01-16
tags: [nlp, hmm, hidden-markov-model, probability, sequence-modeling]
---

# 隐马尔可夫模型 (HMM) 详解

**隐马尔可夫模型 (Hidden Markov Model, HMM)** 是马尔可夫链的扩展，它在普通马尔可夫链的基础上增加了一个关键概念：**隐藏状态**。

简单来说：**你看不到真正的状态，只能通过观测到的线索来推测它。**

---

## 1. 从马尔可夫链到 HMM

### 1.1 普通马尔可夫链

在普通马尔可夫链中，状态是**可见的**：

- 今天是晴天 → 你直接看到了
- 根据转移概率预测明天

### 1.2 隐马尔可夫模型

在 HMM 中，状态是**隐藏的**：

- 你看不到真正的天气（比如你在没有窗户的房间里）
- 你只能看到一些**观测信号**（比如同事带伞了）
- 需要通过这些信号来**推测**隐藏状态

---

## 2. HMM 的五元组

一个完整的 HMM 由以下五个部分组成：

| 符号 | 名称 | 说明 |
|------|------|------|
| $S$ | 隐状态集合 | 所有可能的隐藏状态 |
| $O$ | 观测集合 | 所有可能的观测值 |
| $A$ | 转移概率矩阵 | $a_{ij} = P(s_j \| s_i)$，从状态 i 转移到状态 j 的概率 |
| $B$ | 发射概率矩阵 | $b_j(o) = P(o \| s_j)$，在状态 j 观测到 o 的概率 |
| $\pi$ | 初始概率分布 | 初始时处于各状态的概率 |

---

## 3. 实战案例：天气预测

### 3.1 场景设定

> **场景：** 你被关在一个**没有窗户的房间**里，看不到外面的天。
> **目标：** 预测**明天**的天气。
> **线索：** 每天中午，同事小王从外面走进来，你可以观察他有没有带伞。

### 3.2 模型定义

**隐状态 (Hidden States)：** 外面的真实天气

- $s_1$：晴天
- $s_2$：雨天

**观测值 (Observations)：** 小王的状态

- $o_1$：带伞
- $o_2$：不带伞

**转移概率 A（天气变化规律）：**

|  | 明天晴 | 明天雨 |
|--|--------|--------|
| 今天晴 | 0.7 | 0.3 |
| 今天雨 | 0.4 | 0.6 |

**发射概率 B（小王带伞的习惯）：**

|  | 带伞 | 不带伞 |
|--|------|--------|
| 晴天 | 0.1 | 0.9 |
| 雨天 | 0.8 | 0.2 |

### 3.3 预测流程

#### 步骤 1：收集观测 (Observation)

今天小王走进来，**手里拿着一把滴水的伞** ($o_1$)。

#### 步骤 2：反推当前状态 (Filtering)

**问题：** 小王带伞了，外面的天气是什么？

利用**发射概率 B**：
- 雨天带伞的概率：0.8
- 晴天带伞的概率：0.1

使用贝叶斯定理：

$$
P(\text{雨} | \text{带伞}) = \frac{P(\text{带伞} | \text{雨}) \cdot P(\text{雨})}{P(\text{带伞})}
$$

**推论：** 今天外面大概率是**雨天**。

#### 步骤 3：预测未来状态 (Prediction)

**问题：** 既然今天下雨，明天会怎样？

利用**转移概率 A**：
- 雨天 → 雨天：0.6
- 雨天 → 晴天：0.4

**最终预测：** 明天有 60% 的概率继续下雨。

---

## 4. HMM 的三大问题

### 4.1 评估问题 (Evaluation)

**问题：** 给定模型参数和观测序列，计算该序列出现的概率。

**算法：** 前向算法 (Forward Algorithm) / 后向算法 (Backward Algorithm)

**应用：** 语音识别中判断一段语音属于哪个词

### 4.2 解码问题 (Decoding)

**问题：** 给定模型参数和观测序列，找出最可能的隐状态序列。

**算法：** 维特比算法 (Viterbi Algorithm)

**应用：** 词性标注、命名实体识别

### 4.3 学习问题 (Learning)

**问题：** 给定观测序列，学习模型参数（A, B, π）。

**算法：** Baum-Welch 算法（EM 算法的特例）

**应用：** 从数据中训练 HMM 模型

---

## 5. 代码实现

### 5.1 简单 HMM 实现

```python
import numpy as np

class HMM:
    def __init__(self, states, observations, A, B, pi):
        """
        初始化 HMM

        Args:
            states: 隐状态列表
            observations: 观测值列表
            A: 转移概率矩阵
            B: 发射概率矩阵
            pi: 初始概率分布
        """
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_obs = len(observations)

        self.A = np.array(A)  # 转移概率
        self.B = np.array(B)  # 发射概率
        self.pi = np.array(pi)  # 初始概率

        # 索引映射
        self.state_idx = {s: i for i, s in enumerate(states)}
        self.obs_idx = {o: i for i, o in enumerate(observations)}

    def forward(self, obs_seq):
        """
        前向算法：计算观测序列的概率

        Returns:
            alpha: 前向概率矩阵
            prob: 观测序列的概率
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))

        # 初始化
        obs_0 = self.obs_idx[obs_seq[0]]
        alpha[0] = self.pi * self.B[:, obs_0]

        # 递推
        for t in range(1, T):
            obs_t = self.obs_idx[obs_seq[t]]
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_t]

        prob = np.sum(alpha[-1])
        return alpha, prob

    def viterbi(self, obs_seq):
        """
        维特比算法：找出最可能的隐状态序列

        Returns:
            path: 最可能的状态序列
            prob: 该路径的概率
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # 初始化
        obs_0 = self.obs_idx[obs_seq[0]]
        delta[0] = self.pi * self.B[:, obs_0]

        # 递推
        for t in range(1, T):
            obs_t = self.obs_idx[obs_seq[t]]
            for j in range(self.n_states):
                probs = delta[t-1] * self.A[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) * self.B[j, obs_t]

        # 回溯
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        prob = np.max(delta[-1])

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return [self.states[i] for i in path], prob


# 使用示例：天气模型
hmm = HMM(
    states=['晴', '雨'],
    observations=['带伞', '不带伞'],
    A=[
        [0.7, 0.3],  # 晴 -> [晴, 雨]
        [0.4, 0.6]   # 雨 -> [晴, 雨]
    ],
    B=[
        [0.1, 0.9],  # 晴 -> [带伞, 不带伞]
        [0.8, 0.2]   # 雨 -> [带伞, 不带伞]
    ],
    pi=[0.6, 0.4]  # 初始概率 [晴, 雨]
)

# 观测序列：连续三天小王带伞、不带伞、带伞
obs_seq = ['带伞', '不带伞', '带伞']

# 计算观测序列概率
_, prob = hmm.forward(obs_seq)
print(f"观测序列概率: {prob:.6f}")

# 推测最可能的天气序列
path, path_prob = hmm.viterbi(obs_seq)
print(f"最可能的天气序列: {path}")
print(f"路径概率: {path_prob:.6f}")
```

### 5.2 使用 hmmlearn 库

```python
from hmmlearn import hmm
import numpy as np

# 创建多项式 HMM
model = hmm.CategoricalHMM(n_components=2, n_iter=100)

# 设置模型参数
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
model.emissionprob_ = np.array([
    [0.1, 0.9],  # 晴天
    [0.8, 0.2]   # 雨天
])

# 观测序列 (0=带伞, 1=不带伞)
obs = np.array([[0], [1], [0]])

# 预测隐状态
logprob, states = model.decode(obs, algorithm="viterbi")
print(f"预测状态序列: {states}")  # [1, 0, 1] 对应 [雨, 晴, 雨]
print(f"对数概率: {logprob:.4f}")
```

---

## 6. HMM 的经典应用

### 6.1 词性标注 (POS Tagging)

| 隐状态 | 观测值 |
|--------|--------|
| 词性（名词、动词、形容词...） | 实际的词 |

**例子：**

```
观测: "I    love  cats"
隐状态: PRP   VBP   NNS
       (代词) (动词) (名词复数)
```

### 6.2 语音识别

| 隐状态 | 观测值 |
|--------|--------|
| 音素序列 | 声学特征（MFCC） |

### 6.3 生物信息学

| 隐状态 | 观测值 |
|--------|--------|
| 基因的功能区域 | DNA 碱基序列 |

---

## 7. HMM 的局限性

| 局限 | 说明 |
|------|------|
| **马尔可夫假设** | 当前状态只依赖前一状态，无法建模长距离依赖 |
| **独立输出假设** | 观测只依赖当前状态，忽略观测之间的关系 |
| **离散状态** | 标准 HMM 只能处理离散状态 |
| **局部最优** | Baum-Welch 算法可能陷入局部最优 |

这些局限性促使了更先进模型的发展：
- **CRF (条件随机场)**：解决独立输出假设
- **RNN/LSTM**：解决长距离依赖
- **Transformer**：全局注意力机制

---

## 8. 总结

**HMM 使用说明书：**

1. **看方块 ($O$)**：获取当前的线索（小王带伞）
2. **逆着蓝色箭头 ($B$) 思考**：利用线索推测现在的隐状态是什么（外面在下雨）
3. **顺着红色箭头 ($A$) 思考**：利用现在的状态，根据统计规律，推测下一个状态是什么（明天大概率还下雨）

这就是 HMM 在预测中的核心逻辑：**由表（观测）及里（当前状态），再由里推未来（未来状态）。**

---

## 参考资料

- [Wikipedia: Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
- [Speech and Language Processing - Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition
