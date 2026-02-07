---
title: N-gram 语言模型详解
sidebar_label: N-gram
sidebar_position: 1
date: 2025-01-16
tags: [nlp, n-gram, language-model, markov-assumption, engram]
---

# N-gram 语言模型详解

**N-gram** 是自然语言处理（NLP）和概率论中一个非常经典且基础的概念。简单来说，它是指文本序列中 **N 个连续的项（通常是词或字符）** 组成的序列。

在深度学习（如 Transformer）出现之前，N-gram 曾是语言模型的主流统治者。虽然现在 LLM 是主流，但 N-gram 的思想正在以新的形式回归（如 Engram 架构）。

---

## 1. 什么是 N-gram？

想象你在读一个句子，手中拿着一个"滑动窗口"，这个窗口一次只能框住 N 个词。这个框住的片段就是 N-gram。

假设有一个句子：**"DeepSeek releases a new model"**

### 1.1 Unigram (N=1)

窗口大小为 1：

```
"DeepSeek", "releases", "a", "new", "model"
```

**用途：** 词频统计（这个词出现了多少次）。由于没有上下文，无法预测下一个词。

### 1.2 Bigram (N=2)

窗口大小为 2：

```
"DeepSeek releases", "releases a", "a new", "new model"
```

**用途：** 能够看到前一个词。比如看到 "new"，模型可能会预测下一个词是 "model" 或 "york"。

### 1.3 Trigram (N=3)

窗口大小为 3：

```
"DeepSeek releases a", "releases a new", "a new model"
```

**用途：** 上下文更丰富，预测更准确。

---

## 2. N-gram 的核心逻辑：马尔可夫假设

N-gram 模型的核心思想是：**预测下一个词出现的概率，只需要看它前面 N-1 个词，而不需要看整个句子。**

这被称为**马尔可夫假设 (Markov Assumption)**。

### 2.1 概率公式

对于一个词序列 $w_1, w_2, ..., w_n$，其联合概率可以表示为：

$$
P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})
$$

在 N-gram 模型中，我们近似为：

$$
P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-(N-1)}, ..., w_{i-1})
$$

### 2.2 实际例子

如果我们要预测 "I love artificial [?]"：

- 如果是 **Bigram**，它只看 "artificial"，可能会预测 "flower"（假花）或 "intelligence"（人工智能）
- 如果是 **Trigram**，它会看 "love artificial"，那么预测 "intelligence" 的概率就会大大增加

---

## 3. N-gram 概率计算

### 3.1 最大似然估计 (MLE)

N-gram 概率通过统计语料库中的出现频率来估计：

$$
P(w_n | w_{n-N+1}, ..., w_{n-1}) = \frac{C(w_{n-N+1}, ..., w_n)}{C(w_{n-N+1}, ..., w_{n-1})}
$$

其中 $C(\cdot)$ 表示序列在语料库中的出现次数。

**例子：** 假设语料库中：
- "a new" 出现了 100 次
- "a new model" 出现了 20 次

那么 $P(\text{model} | \text{a new}) = \frac{20}{100} = 0.2$

### 3.2 平滑技术

为了解决零概率问题（某些 N-gram 从未在训练集中出现），常用的平滑技术包括：

1. **加一平滑 (Laplace Smoothing)**：给每个计数加 1
2. **Good-Turing 平滑**：用低频词的频率来估计未见词的概率
3. **Kneser-Ney 平滑**：考虑词在不同上下文中出现的多样性

---

## 4. N-gram 的优缺点

### 4.1 优点

**速度极快：**
- 它是查表或简单的统计，计算复杂度通常是 $O(1)$
- 不需要像 Transformer 那样进行复杂的矩阵乘法
- 这也是为什么 Engram 论文选择它来实现"条件记忆"的原因

**捕捉局部模式：**
- 对于固定短语（如 "New York City"）非常有效
- 适合处理成语（"四大发明"）和习惯用语
- 这些词总是粘在一起出现的

### 4.2 缺点

**长距离依赖缺失：**
- 随着 N 增大，需要存储的组合数量呈指数级爆炸（维度灾难）
- 通常 N 最多只能取到 4 或 5
- 无法理解长距离的上下文（例如文章开头的伏笔，结尾才呼应）

**稀疏性问题：**
- 如果训练语料里从来没出现过 "eat a computer" 这个搭配
- N-gram 模型就会认为这句话的概率是 0
- 虽然语法上它是对的

---

## 5. N-gram 在现代 AI 中的应用

### 5.1 Engram 架构：N-gram 的复兴

在 Transformer 时代，N-gram 看起来像是一项过时的技术，但 Engram 论文对它进行了**现代化改造**，解决了它在现代大模型中的痛点：

#### 从"统计概率"变成"向量嵌入"

传统的 N-gram 存的是概率（数字），而 Engram 存的是 **Embedding（向量）**。这意味着它把 N-gram 变成了一种可以被神经网络理解的语义表示。

#### 用哈希解决"存储爆炸"

为了避免存储所有可能的词组，Engram 使用了**多头哈希 (Multi-Head Hashing)**。通过哈希函数将 N-gram 映射到固定大小的表中，大大节省了空间，使得它可以扩展到 100B 甚至更大的参数规模。

#### 弥补 Transformer 的短板

- Transformer 擅长长距离推理（算力密集）
- N-gram 擅长短语死记硬背（存储密集）
- 论文证明了用 N-gram 处理局部短语，能让 Transformer 腾出手来处理更难的任务
- 这是一种完美的互补

### 5.2 其他应用场景

| 应用 | 说明 |
|------|------|
| 拼写检查 | 检测不常见的词组合 |
| 文本生成 | 简单场景下的快速生成 |
| 语音识别 | 语言模型打分 |
| 机器翻译 | 流畅度评估 |
| 信息检索 | 文档相似度计算 |

---

## 6. 代码实现

### 6.1 Python 简单实现

```python
from collections import defaultdict, Counter

class NgramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()

    def train(self, corpus):
        """训练 N-gram 模型"""
        for sentence in corpus:
            # 添加起始和结束标记
            tokens = ['<s>'] * (self.n - 1) + sentence.split() + ['</s>']

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]

                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

    def probability(self, context, word):
        """计算给定上下文的词概率"""
        context = tuple(context)
        if self.context_counts[context] == 0:
            return 0.0
        return self.ngram_counts[context][word] / self.context_counts[context]

    def predict_next(self, context, top_k=5):
        """预测下一个词"""
        context = tuple(context[-(self.n-1):])
        candidates = self.ngram_counts[context]
        return candidates.most_common(top_k)


# 使用示例
corpus = [
    "I love natural language processing",
    "I love machine learning",
    "natural language processing is fun",
    "machine learning is powerful"
]

model = NgramModel(n=2)  # Bigram
model.train(corpus)

# 预测
print(model.predict_next(['I']))  # [('love', 2)]
print(model.probability(['I'], 'love'))  # 1.0
```

### 6.2 使用 NLTK

```python
import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# 准备数据
text = [['I', 'love', 'NLP'], ['NLP', 'is', 'fun']]
n = 2

# 创建训练数据
train_data, padded_sents = padded_everygram_pipeline(n, text)

# 训练模型
model = MLE(n)
model.fit(train_data, padded_sents)

# 查询概率
print(model.score('love', ['I']))  # P(love|I)
```

---

## 7. 总结

N-gram 本质上就是**语言的"惯用搭配库"**。虽然在深度学习时代它看起来有些简单，但其思想依然具有重要价值：

1. **简单高效**：查表操作，速度极快
2. **局部模式**：擅长捕捉固定搭配和短语
3. **现代复兴**：在 Engram 等新架构中以向量化形式回归

在现代架构（如 Engram）中，N-gram 充当了模型的**外部硬盘**，存储了大量固定的知识块，供模型随取随用，与 Transformer 形成完美互补。

---

## 参考资料

- [Speech and Language Processing - N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [NLTK Language Modeling](https://www.nltk.org/api/nltk.lm.html)
- Engram: Speculative Sampling Meets N-gram Memory
