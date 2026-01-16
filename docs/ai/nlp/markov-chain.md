---
title: 马尔可夫链详解
sidebar_label: 马尔可夫链
sidebar_position: 2
date: 2025-01-16
tags: [nlp, markov-chain, probability, language-model, markov-property]
---

# 马尔可夫链详解

**马尔可夫链 (Markov Chain)** 是概率论和数理统计中一个非常核心的模型。它以俄罗斯数学家安德雷·马尔可夫（Andrey Markov）的名字命名。

简单一句话概括：**"明天会发生什么，只取决于今天，而与昨天以及过去无关。"**

这被称为**无记忆性 (Memorylessness)**，是马尔可夫链的灵魂。

---

## 1. 核心概念：无记忆性 (Markov Property)

在现实生活中，很多事情是有因果积累的（比如你今天累是因为前天熬夜了）。但在马尔可夫链的数学假设中，我们切断了这种长期的因果线。

### 1.1 数学定义

假设有一个序列 $X_1, X_2, ..., X_n$。

要预测下一时刻 $X_{n+1}$ 的状态，只需要知道当前时刻 $X_n$ 的状态。之前所有的历史状态 $X_1, ..., X_{n-1}$ 对未来都没有额外影响。

$$
P(X_{n+1} | X_1, X_2, ..., X_n) = P(X_{n+1} | X_n)
$$

### 1.2 通俗理解

就像下棋（围棋或象棋）：

- **非马尔可夫视角：** 复盘整局棋，看你是怎么一步步走到现在的
- **马尔可夫视角：** 只看**现在的棋盘**。不管你之前走了什么臭棋或妙手，下一步的最佳走法只取决于**当前棋盘的局面**

当前的局面包含了过去所有的信息沉淀。

---

## 2. 组成部分：状态与转移

一个标准的马尔可夫链包含两个核心要素：

### 2.1 状态空间 (State Space)

所有可能出现的情况。

**例子：**
- 天气（晴、雨）
- 股票（涨、跌）
- 心情（好、坏）

### 2.2 转移概率 (Transition Probabilities)

从一个状态跳到另一个状态的可能性。

---

## 3. 经典案例：天气模型

假设天气只有两种状态：**晴天 (Sunny)** 和 **雨天 (Rainy)**。

- 如果今天是晴天，明天有 90% 是晴天，10% 是雨天
- 如果今天是雨天，明天有 50% 是晴天，50% 是雨天

这就构成了一个马尔可夫链。

### 3.1 状态转移图

```
         0.9
    ┌──────────┐
    │          │
    ▼          │
  ┌───┐  0.5  ┌───┐
  │晴天│◄─────│雨天│
  └───┘       └───┘
    │          ▲
    │   0.1    │
    └──────────┘
         0.5
         │
         └──────┐
                │
                ▼
```

### 3.2 转移矩阵

我们可以把转移概率写成一个**转移矩阵**：

|  | 明天晴 | 明天雨 |
|--|--------|--------|
| **今天晴** | 0.9 | 0.1 |
| **今天雨** | 0.5 | 0.5 |

用数学符号表示：

$$
P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}
$$

只要有这个矩阵和今天的状态，你就可以算出后天、大后天甚至一万年后的天气概率分布。

### 3.3 多步转移

如果今天是晴天，想知道**后天**的天气分布，只需要计算 $P^2$：

$$
P^2 = P \times P = \begin{pmatrix} 0.86 & 0.14 \\ 0.70 & 0.30 \end{pmatrix}
$$

所以后天是晴天的概率是 86%。

---

## 4. 马尔可夫链与 N-gram 的关系

在自然语言处理（NLP）中，**N-gram 模型本质上就是 N-1 阶的马尔可夫链**。

### 4.1 1 阶马尔可夫链 (Bigram)

下一个词只取决于前 1 个词。

**例子：** "Artificial" → "Intelligence" (看 "Artificial" 猜 "Intelligence")

$$
P(w_n | w_1, ..., w_{n-1}) \approx P(w_n | w_{n-1})
$$

### 4.2 2 阶马尔可夫链 (Trigram)

下一个词只取决于前 2 个词。

**例子：** "I love" → "you" (看 "I love" 猜 "you")

$$
P(w_n | w_1, ..., w_{n-1}) \approx P(w_n | w_{n-2}, w_{n-1})
$$

这里的 N-1 阶，就是把"当前状态"的定义扩大了。在 2 阶链中，"当前状态"不再是 1 个词，而是"2 个词的组合"。

---

## 5. 为什么它在大模型时代"不够用"了？

虽然马尔可夫链简单高效（也是 Engram 论文中用来做快速查表的基础），但它在处理人类语言时有致命缺陷：**长距离依赖 (Long-term Dependencies)**。

### 5.1 经典问题案例

> "The **girl** who lost her keys while running in the park ...... was **sad**."
> (那个在公园跑步时丢了钥匙的**女孩**......很**伤心**。)

**马尔可夫链（比如 Bigram）的视角：**
- 看到 "was" 预测下一个词
- 它只看前面一个词 "park" 或者 "running"
- 它早就**忘**了这句话的主语是 "girl"

**现实需求：**
- 要填出 "sad"（伤心），模型必须记得很久之前的 "girl" 和 "lost keys"

### 5.2 解决方案演进

马尔可夫链因为其"无记忆性"（或者只有极短的 N-gram 记忆），无法跨越这么长的距离去建立联系。

| 模型 | 特点 | 记忆能力 |
|------|------|----------|
| 马尔可夫链 | 只看前 N-1 个词 | 极短 |
| RNN | 循环结构传递信息 | 中等（梯度消失问题） |
| LSTM/GRU | 门控机制 | 较长 |
| **Transformer** | 注意力机制 | **任意长度** |

Transformer（注意力机制）打破了马尔可夫假设，它允许模型在预测下一个词时，**回头看全文**，关注任何位置的重要信息。

---

## 6. 代码实现

### 6.1 简单马尔可夫链

```python
import numpy as np

class MarkovChain:
    def __init__(self, states, transition_matrix):
        """
        初始化马尔可夫链

        Args:
            states: 状态列表，如 ['晴', '雨']
            transition_matrix: 转移概率矩阵
        """
        self.states = states
        self.n_states = len(states)
        self.transition_matrix = np.array(transition_matrix)

        # 创建状态到索引的映射
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.idx_to_state = {i: s for i, s in enumerate(states)}

    def next_state(self, current_state):
        """根据当前状态预测下一个状态"""
        idx = self.state_to_idx[current_state]
        probs = self.transition_matrix[idx]
        next_idx = np.random.choice(self.n_states, p=probs)
        return self.idx_to_state[next_idx]

    def simulate(self, start_state, n_steps):
        """模拟 n 步"""
        states_sequence = [start_state]
        current = start_state

        for _ in range(n_steps):
            current = self.next_state(current)
            states_sequence.append(current)

        return states_sequence

    def steady_state(self):
        """计算稳态分布"""
        # 求解 πP = π，即 (P^T - I)π = 0
        A = self.transition_matrix.T - np.eye(self.n_states)
        A = np.vstack([A, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1

        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        return {self.idx_to_state[i]: p for i, p in enumerate(pi)}


# 使用示例：天气模型
weather_chain = MarkovChain(
    states=['晴', '雨'],
    transition_matrix=[
        [0.9, 0.1],  # 晴天 -> [晴, 雨]
        [0.5, 0.5]   # 雨天 -> [晴, 雨]
    ]
)

# 模拟 10 天天气
print("10天天气模拟:", weather_chain.simulate('晴', 10))

# 计算稳态分布
print("稳态分布:", weather_chain.steady_state())
# 输出: {'晴': 0.833, '雨': 0.167}
```

### 6.2 文本生成

```python
from collections import defaultdict
import random

class TextMarkovChain:
    def __init__(self, order=1):
        self.order = order
        self.transitions = defaultdict(list)

    def train(self, text):
        """从文本学习转移概率"""
        words = text.split()

        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            self.transitions[state].append(next_word)

    def generate(self, start_words, length=20):
        """生成文本"""
        if len(start_words) != self.order:
            raise ValueError(f"需要 {self.order} 个起始词")

        result = list(start_words)
        current = tuple(start_words)

        for _ in range(length):
            if current not in self.transitions:
                break
            next_word = random.choice(self.transitions[current])
            result.append(next_word)
            current = tuple(result[-self.order:])

        return ' '.join(result)


# 使用示例
corpus = """
I love machine learning and deep learning.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks.
I love artificial intelligence.
"""

model = TextMarkovChain(order=2)
model.train(corpus)

print(model.generate(['I', 'love'], length=10))
```

---

## 7. 总结

| 特性 | 描述 |
|------|------|
| **核心思想** | 无记忆性：未来只取决于现在 |
| **组成要素** | 状态空间 + 转移概率矩阵 |
| **与 N-gram 关系** | N-gram 是 N-1 阶马尔可夫链 |
| **优点** | 简单、计算快 |
| **缺点** | 无法捕捉长距离依赖 |
| **现代角色** | Transformer 的辅助（如 Engram 模块） |

马尔可夫链是一个"活在当下"的数学模型，虽然在需要长上下文理解的 AI 时代它退居二线，但作为 Transformer 的辅助模块（如 Engram），它依然发挥着重要作用。

---

## 参考资料

- [Wikipedia: Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)
- [Speech and Language Processing - N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- Engram: Speculative Sampling Meets N-gram Memory
