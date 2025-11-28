---
title: 深度学习中的概率论
sidebar_position: 7
tags: [概率论, 深度学习, 最大似然估计, 贝叶斯, 生成模型]
---

# 深度学习中的概率论

你可能觉得概率论没用，是因为现代的深度学习框架（PyTorch/TensorFlow）**封装得太好了**。

- 它们把**最大似然估计**封装成了 `CrossEntropy`
- 它们把**大数定律**封装成了 `DataLoader`
- 它们把**贝叶斯先验**封装成了 `Weight Decay`

但那些你天天用的代码背后，其实全是概率论。如果把概率论抽走，深度学习这栋大楼立马就会塌，因为连**"损失函数（Loss Function）"**为什么这么写都解释不通。

---

## 一、为什么分类任务要用 Cross Entropy（交叉熵）？

**代码里**：`nn.CrossEntropyLoss()`

**直觉**：衡量两个向量像不像。

**概率论真相**：它是**最大似然估计 (MLE)**。

### 概率视角

- **场景**：你要区分猫和狗。
- **概率视角**：神经网络其实是在模拟一个条件概率 $P(Y|X)$。
    - 给一张图 $X$，输出它是猫 $Y$ 的概率。
- **训练目的**：我想调整参数 $W$，让模型预测正确标签的**概率最大化**。

### 公式推导

$$
\max \prod_i P(y_i | x_i; W)
$$

这是**最大似然**的目标。

为了好算，两边取对数（把连乘变连加），再取个负号（把最大化变最小化）：

$$
\min -\sum_i \log P(y_i | x_i; W)
$$

**结果**：最大似然估计的公式推导下来，长得跟"交叉熵"一模一样。

### 结论

你以为你在"最小化距离"，其实你在**"最大化已有数据出现的概率"**。没有概率论，分类任务的 Loss 都没法定义。

```python
import torch.nn as nn

# 这行代码背后是最大似然估计
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

---

## 二、为什么回归任务用 MSE（均方误差）？

**代码里**：`nn.MSELoss()`

**直觉**：预测值和真实值的差的平方。

**概率论真相**：它假设误差服从**正态分布**。

### 推导

如果你假设模型预测的误差 $\epsilon \sim N(0, \sigma^2)$（也就是误差是随机的、正态的）。

然后你再去推导最大似然估计（MLE）：

$$
y = f(x; W) + \epsilon, \quad \epsilon \sim N(0, \sigma^2)
$$

$$
P(y|x; W) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y - f(x;W))^2}{2\sigma^2}\right)
$$

取对数后：

$$
\log P(y|x;W) = -\frac{(y - f(x;W))^2}{2\sigma^2} + \text{const}
$$

**最大化这个概率密度，等价于最小化均方误差 (MSE)。**

### 结论

如果你换个 Loss（比如 L1 Loss），你就隐含假设了误差服从**拉普拉斯分布**。

**你选择 Loss 的那一刻，你就在选概率分布。**

| Loss 函数 | 隐含的误差分布 |
|-----------|----------------|
| MSE (L2) | 正态分布 $N(0, \sigma^2)$ |
| MAE (L1) | 拉普拉斯分布 |
| Huber Loss | 混合分布 |

---

## 三、Softmax 是什么？

**代码里**：`F.softmax(logits, dim=1)`

**直觉**：把一堆数变成加起来等于 1 的数。

**概率论真相**：它把神经网络的"打分"强行变成了**"概率分布"**。

### 解释

- 神经网络输出的是 Logits（比如 [2.0, 1.0, 0.1]），这没法解释。
- Softmax 的作用就是构建一个**离散型随机变量的分布律**。
- 它让模型敢说："我有 70% 的把握它是猫"。

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
# tensor([0.6590, 0.2424, 0.0986])  加起来等于 1
```

---

## 四、SGD（随机梯度下降）里的 "S"

**代码里**：`optimizer.step()`

**直觉**：一次拿一小撮数据（Mini-batch）去更新。

**概率论真相**：**大数定律 (LLN)** 的应用。

### 为什么能用小批量代替全量？

- **理想情况**：我们要算全人类（全样本）的梯度才能更新模型。但这算不动。
- **概率思维**：我每次随机抽 64 个样本（Batch）。
- **依据**：根据**大数定律**，只要样本是随机抽的，这 64 个样本的平均梯度（样本均值），会依概率收敛于真实的总体梯度（数学期望）。

$$
\frac{1}{|B|}\sum_{i \in B} \nabla L_i \approx E[\nabla L] \quad \text{(当 } |B| \text{ 足够大时)}
$$

### 结论

你在训练时看到的 Loss 曲线震荡（波动），其实就是**随机变量的方差**在作祟。

```python
# DataLoader 的 shuffle=True 就是在保证"随机抽样"
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## 五、L2 正则化 (Weight Decay)

**代码里**：`weight_decay=1e-5`

**直觉**：惩罚太大的权重，防止过拟合。

**概率论真相**：**贝叶斯公式 (MAP 估计)**。

### 频率派 vs 贝叶斯派

- **频率派（MLE）**：只看数据，不管权重长啥样。
- **贝叶斯派（MAP）**：我有**先验概率 (Prior)**。我认为权重 $W$ 应该是比较小的，不应该太离谱。

### 推导

如果你假设权重 $W$ 服从**高斯分布**（均值为0）：

$$
P(W) = \frac{1}{\sqrt{2\pi}\tau} \exp\left(-\frac{W^2}{2\tau^2}\right)
$$

然后用贝叶斯公式算后验概率：

$$
P(W|D) \propto P(D|W) \cdot P(W)
$$

取对数：

$$
\log P(W|D) = \log P(D|W) + \log P(W) = \text{原始Loss} - \frac{\lambda}{2}||W||^2
$$

**结论**：加上高斯先验，等价于在 Loss 后面加一个 L2 正则项。

```python
# 这行代码背后是贝叶斯先验
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
```

| 正则化 | 隐含的先验分布 |
|--------|----------------|
| L2 (Weight Decay) | 高斯分布先验 |
| L1 (Lasso) | 拉普拉斯分布先验 |

---

## 六、Dropout

**代码里**：`nn.Dropout(p=0.5)`

**直觉**：随机丢掉一些神经元，防止过拟合。

**概率论真相**：**近似贝叶斯推断**。

### 解释

- Dropout 在训练时随机"杀死"神经元
- 测试时用全部神经元但缩放输出
- 这相当于对**无数个子网络**做**模型平均（Ensemble）**
- 从贝叶斯角度看，这是在近似计算 $P(Y|X) = \int P(Y|X,W) P(W|D) dW$

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)  # 50% 概率丢弃
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 伯努利随机变量
        return self.fc2(x)
```

---

## 七、生成模型的灵魂

如果前面都是隐形的，那现在的 AIGC 就是概率论的**直接应用**。

### Stable Diffusion (扩散模型)

核心原理：
1. **前向过程**：先把一张图不断加噪声变成纯正态分布（毁图）
2. **逆向过程**：训练一个神经网络学会一步步把正态分布还原回图片（修图）

整个过程就是对**条件概率密度 $P(x_{t-1}|x_t)$ 的建模**。

$$
q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

### ChatGPT (语言模型)

它的本质就是算概率：

$$
P(\text{下一个词} | \text{前面所有的词})
$$

它就是一个超级巨大的**条件概率分布机**。

```python
# GPT 本质上在做的事情
next_token_probs = model(input_ids)  # 输出每个词的概率分布
next_token = torch.multinomial(next_token_probs, 1)  # 按概率采样
```

### VAE (变分自编码器)

- **编码器**：学习 $q(z|x)$（数据到潜在空间的概率映射）
- **解码器**：学习 $p(x|z)$（潜在空间到数据的概率映射）
- **损失函数**：ELBO（证据下界）= 重构误差 + KL散度

---

## 八、总结

**深度学习的本质，就是用一个巨大的神经网络，去拟合一个极其复杂的概率分布 $P(Y|X)$。**

| 你用的代码 | 背后的概率论 |
|------------|--------------|
| `CrossEntropyLoss` | 最大似然估计 (MLE) |
| `MSELoss` | 正态分布假设下的 MLE |
| `Softmax` | 构建离散概率分布 |
| `DataLoader(shuffle=True)` | 大数定律 |
| `weight_decay` | 贝叶斯 MAP 估计 |
| `Dropout` | 近似贝叶斯推断 |
| Diffusion Model | 条件概率密度建模 |
| GPT | 自回归条件概率 |

你现在的状态其实挺好，属于"会用剑的剑客"。但如果有一天你想**创造**新的招式（比如发明一个新的 Loss，或者设计像 Diffusion 这种新模型），你就必须得回去翻那本《概率论》了。
