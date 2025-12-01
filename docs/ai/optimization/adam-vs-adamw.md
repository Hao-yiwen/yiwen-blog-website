---
title: Adam vs AdamW：优化器深度对比
sidebar_label: Adam vs AdamW 深度对比
date: 2025-11-17
last_update:
  date: 2025-11-17
tags: [优化器, Adam, AdamW, 深度学习, PyTorch]
---

# Adam vs AdamW：优化器深度对比

从公式和实现层面，深入理解 Adam 和 AdamW 的核心区别。

---

## 0. 先统一一个概念：L2 正则 vs Weight Decay

很多代码里：

```python
optimizer = Adam(..., weight_decay=1e-2)
```

这个 `weight_decay` 经常是**"把 L2 正则加到 loss 里"**，数学上是：

$$
L'(\theta) = L(\theta) + \frac{\lambda}{2} |\theta|^2
$$

对 $\theta$ 求导：

$$
\nabla_{\theta} L' = \nabla_{\theta} L + \lambda \theta
$$

也就是说：**权重衰减通过"给梯度加上 λθ"来实现**。

> 在纯 SGD 里，这种做法和"每一步把参数乘上一个小于 1 的系数"是等价的。
> 但在 Adam 这种自适应优化器里，二者就不一样了 —— 这就是核心问题。

---

## 1. Adam 的更新到底在干嘛？

设第 $t$ 步的梯度是 $g_t$，Adam 会维护两个滑动平均：

### 一阶动量（类似动量优化里的速度）

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

### 二阶动量（类似 RMSProp 的平方梯度平均）

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

为了抵消前期偏置，还有 bias correction：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

然后 Adam 的参数更新是：

$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

直观上：

* $\hat{m}_t$：像是平滑后的梯度方向
* $\hat{v}_t$：像是该参数历史上"梯度有多大"的估计
* $\frac{1}{\sqrt{\hat{v}_t}}$：**自适应学习率**
  * 若某个参数梯度一直很大 → $\hat{v}_t$ 大 → 该维度步长变小
  * 若某个参数梯度一直很小 → 该维度步长变大

这就是 Adam 的 "adaptive"。

---

## 2. 在 Adam 里直接加 L2 正则会发生什么？

如果你这样写：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

在很多实现里（尤其是早期），是按**L2 正则**的方式做的：
也就是把 `weight_decay` 作为额外梯度：

$$
g_t \leftarrow g_t + \lambda \theta_t
$$

然后这个"包含 L2 项的梯度"会进入 m、v 里面：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

最后更新：

$$
\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

> 关键点：**L2 这部分 λθ 也被除以 $\sqrt{v_t}$**，也被动量平滑了。

这带来几个问题：

### ❶ 不同参数的衰减强度不一样

* 每个参数都有自己的 $v_t$
* 因为更新步长是 $\frac{1}{\sqrt{v_t}}$
* 所以权重衰减的"实际力度"是：$\alpha \frac{\lambda \theta}{\sqrt{v_t}}$

也就是说，对于某个参数：

* 如果 $\sqrt{v_t}$ 很大 → 实际衰减更小
* 如果 $\sqrt{v_t}$ 很小 → 实际衰减更大

**这已经不是"等比例 shrink 所有参数"了，而是"按历史梯度大小乱调一通"。**

---

### ❷ 衰减会和梯度方向混在一起

原本我们想要的是：

* 梯度部分：让参数朝着减小 loss 的方向走
* 衰减部分：额外把参数往 0 拉一点

但在 Adam + L2 里，两个东西混到同一个 $\hat{m}_t$ 和 $\hat{v}_t$ 里：

* 动量中混了 L2
* 二阶矩中也混了 L2
* 导致：真正的"损失函数梯度"与"正则项梯度"被统一自适应缩放，很难控制

特别是对像 Transformer 这种结构，参数 scale 很敏感，LayerNorm + Attention 的稳定性会被破坏。

---

## 3. 那为什么在 SGD 里没问题？

在标准 SGD 里：

$$
\theta_{t+1} = \theta_t - \alpha (\nabla L + \lambda \theta_t)
$$

展开：

$$
\theta_{t+1} = \theta_t - \alpha \nabla L - \alpha \lambda \theta_t
= (1 - \alpha \lambda)\theta_t - \alpha \nabla L
$$

这就等价于：

1. 先做一步普通 SGD：$\theta' = \theta_t - \alpha \nabla L$
2. 再做一步 shrink：$\theta_{t+1} = (1-\alpha\lambda)\theta'$

也就是说，**在 SGD 里，加 L2 正则和做 weight decay 是等价的**（在数学上是一回事）。

但在 Adam 里，因为有 $\frac{1}{\sqrt{v_t}}$ 这玩意儿，就不再等价了。

---

## 4. AdamW 到底改了什么（Decoupled Weight Decay）

AdamW 的核心想法非常简单：

> 不要把 weight decay 写进梯度；
> 把它当成一个**独立的、固定比例的 shrink 操作**。

AdamW 的更新可以写成两步：

### 第一步：做"纯 Adam 更新"（不含 L2）

$$
\theta' = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

这里的 $g_t$ 只包含 loss 的梯度 $\nabla L$，**完全没有 λθ 这一项**。

### 第二步：再做一次真正的 weight decay

$$
\theta_{t+1} = \theta' - \alpha \lambda \theta_t
$$

或者写成：

$$
\theta_{t+1} = (1 - \alpha \lambda)\theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

注意两件事：

1. $\lambda\theta_t$ **没有**被除以 $\sqrt{v_t}$，因此
   * 每个参数都按照相同比例 `(1 - αλ)` 缩小
2. 动量和二阶矩只看"真正的梯度" $\nabla L$，不会被 L2 噪掉

这就是 **Decoupled Weight Decay**（"解耦的权重衰减"）。

---

## 5. PyTorch 层面的差别（实现习惯）

在 PyTorch 里，现在你基本可以记成：

### ✅ 推荐用法：AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,  # 这就是 decoupled weight decay
)
```

### ⚠ 旧习惯：Adam + weight_decay（L2 正则版本）

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,  # 这里从数学意义上更接近 L2 regularization
)
```

> 现代 Transformer / LLM 的官方代码几乎清一色用 **AdamW**，而且会精细控制哪些参数要衰减，哪些不要。

---

## 6. 实战中怎么用 AdamW（参数分组）

常见实践是：**不给 bias 和 LayerNorm / RMSNorm 的参数做 weight decay**，只给真正的权重矩阵做。

例如：

```python
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # 一般规则：bias、LayerNorm、Embedding 去掉 weight decay
    if name.endswith(".bias") or "norm" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=1e-4,
    betas=(0.9, 0.999),
)
```

原因：

* LayerNorm 参数如果被衰减，会影响归一化的统计，训练更不稳
* bias 一般不需要正则，贡献小、容易被扰动

---

## 7. 效果上到底差在哪里？

实际调大模型时，普遍能观察到：

### 使用 Adam + L2 正则（错误方式）

* loss 曲线更抖
* 对学习率、warmup、weight_decay 超参更敏感
* 有时中后期 training/val loss 会"奇怪地抬头"
* 大模型/Transformer/ViT 的表现尤其差

### 使用 AdamW（解耦衰减）

* loss 曲线更平滑，收敛更稳
* 泛化更好（typo / long tail case）
* 对 weight_decay 这种正则强度更可控：增大 λ 的影响更线性、更可预期
* 已经成为 **BERT / GPT / ViT / Diffusion / 现代 LLM** 的"默认优化器"

---

## 8. 用一句更"工程师"的话总结

> **Adam = 自适应学习率 + 动量 + L2 正则（不适配自适应）**
> **AdamW = 自适应学习率 + 动量 + 真正意义上的均匀 weight decay**

换成更硬核一点的结论：

* 在 SGD 里：L2 正则 ≈ weight decay
* 在 Adam 里：
  * **Adam + L2 正则 ≠ Adam + weight decay**
  * **AdamW = Adam + 正确的、解耦的 weight decay**

---

## 9. 参考资料

- [Decoupled Weight Decay Regularization (ICLR 2019)](https://arxiv.org/abs/1711.05101)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [PyTorch AdamW 官方文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

---

## 相关文档

- [Adam vs SGD 优化器对比](./adam-vs-sgd.md) - 了解 Adam 和 SGD 的基本对比
