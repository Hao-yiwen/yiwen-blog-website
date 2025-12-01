---
title: 常用学习率调度器简介
sidebar_label: 学习率调度器
date: 2025-11-18
last_update:
    date: 2025-11-18
---

# 常用学习率调度器简介

## 0. 为啥需要学习率调度器?

-   学习率（lr）太大：训练抖动、发散；
-   学习率太小：收敛很慢、容易卡在次优点。
-   **经验：训练早期 lr 大一点多探索，后期 lr 小一点细调。**

学习率调度器（learning rate scheduler）就是：

> 在训练过程中，按照某种策略自动调整 optimizer 里的 `lr`。

在 PyTorch 中，它们都在：`torch.optim.lr_scheduler`。

---

## 1. 固定学习率 / 手动调 lr

### 思想

-   训练全程 `lr` 不变，或者你自己看 loss 曲线、手动改 lr。

### 使用场景

-   小模型、小数据集、debug 阶段；
-   或者你懒得管 / 想先粗跑一版。

### PyTorch 示例

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 不用 scheduler
```

优点：简单粗暴。
缺点：后期不收缩 lr，效果往往不如退火策略。

---

## 2. StepLR / MultiStepLR —— 阶梯下降

### 思想

按「楼梯形」下降 lr：

-   **StepLR**：每隔 `step_size` 个 epoch，把 lr 乘以 `gamma`。
-   **MultiStepLR**：在指定的若干 epoch（milestones）时，把 lr 乘以 `gamma`。

### 公式（StepLR）

$$
lr_t = lr_0 \cdot \gamma^{\lfloor \frac{t}{\text{step_size}} \rfloor}
$$

### PyTorch 示例

```python
# 每 30 个 epoch，lr *= 0.1
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1,
)

# 或者多点阶梯下降
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1,
)
```

### 特点

-   优点：经典、可控、ImageNet 传统配置很多这么写；
-   缺点：lr 变化是「瞬间跳变」，曲线不够平滑。

---

## 3. ExponentialLR —— 指数衰减

### 思想

每个 step / epoch，都乘一个固定比例 $\gamma < 1$。

$$
lr_t = lr_0 \cdot \gamma^t
$$

### PyTorch

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95,  # 每个 epoch lr *= 0.95
)
```

### 特点

-   优点：连续平滑、实现简单；
-   缺点：前期下降较快，后期尾巴较长（lr 会一直往 0 靠近但不为 0）。

---

## 4. CosineAnnealingLR —— 余弦退火（半个余弦波）

### 思想

用一条「半个余弦波」从高 lr 平滑退到低 lr：

$$
lr(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
\Big(1 + \cos(\pi \frac{t}{T_{\max}})\Big)
$$

-   $t$：调用 `scheduler.step()` 的次数（通常是 epoch）
-   $T_{\max}$：从 $\eta_{\max}$ 退火到 $\eta_{\min}$ 的总步数
-   $\eta_{\max}$：初始 lr
-   $\eta_{\min}$：最小 lr

### PyTorch

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,  # 常见用法：= 总 epoch 数
    eta_min=1e-6
)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### 特点

-   形状：两头平、中间陡，非常平滑；
-   早期保持较大 lr，后期贴近 $\eta_{\min}$ 精调；
-   现在各种 CV / NLP 里非常常见，尤其配合 warmup。

---

## 5. CosineAnnealingWarmRestarts —— 余弦 + 周期重启

### 思想

在余弦退火的基础上，每隔一段时间「重启」一次 lr：

-   每个周期内：从 $\eta_{\max}$ → 余弦退火 → $\eta_{\min}$
-   周期结束时：lr 突然重置为 $\eta_{\max}$，再来下一段半个余弦波

### PyTorch

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,     # 第一个周期 10 个 epoch
    T_mult=2,   # 每次周期长度乘 2：10 -> 20 -> 40 ...
    eta_min=1e-6
)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

### 特点

-   有点像周期性「再探索」：每次重启给一个大 lr；
-   理论上有助于跳出局部最优；
-   曲线是好多段平滑的小山丘。

---

## 6. ReduceLROnPlateau —— 指标不涨就砍 lr

### 思想

不是按 epoch 或 step 机械调整，而是看某个 metric（比如 val loss / val acc）：

-   如果若干个 epoch 内指标没有改善，就把 lr 乘以 `factor`。

### PyTorch

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控的指标是越小越好（如 val_loss）
    factor=0.1,        # lr *= 0.1
    patience=5,        # 5 个 epoch 没提升就砍 lr
    min_lr=1e-6,
)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)   # 注意：这里要传入 metric
```

### 特点

-   数据驱动：根据模型表现来决定是否减 lr；
-   常见于传统深度学习项目、Kaggle 代码；
-   和其他 scheduler 可以组合使用（但要小心冲突）。

---

## 7. OneCycleLR —— 一次 cycle，前升后降

### 思想

-   先从较小 lr 升到一个较大的 lr（探索）；
-   然后再退火降到非常小的 lr（收敛）。
    常用于 **大 batch 训练 / fast.ai 风格训练**。

### PyTorch 简化例子

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    total_steps=num_epochs * steps_per_epoch
)

for epoch in range(num_epochs):
    for batch in train_loader:
        train_step(...)
        scheduler.step()   # 每个 iteration 调一次
```

特点：

-   对于一些任务，收敛速度很快；
-   需要事先知道 total_steps，配置稍微麻烦点。

---

## 8. Warmup + 主调度器（非常常见）

### 思想

-   刚开始几百 / 几千 step，lr 从很小**线性升到目标 lr**（warmup）；
-   然后再接普通 scheduler（如 Cosine、MultiStep 等）。

原因：

-   大模型 + AdamW/Adam 直接给大 lr，前期不稳定；
-   warmup 可以让优化更平稳，特别是 Transformer / LLM。

简单伪代码（真实项目一般自己写个 wrapper 或用库）：

```python
def get_lr(it):
    if it < warmup_steps:
        return base_lr * it / warmup_steps  # 线性升
    # 之后走余弦退火
    t = it - warmup_steps
    T = total_steps - warmup_steps
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / T))
```

---

## 9. 如何选?

一个非常实用的「懒人选择」：

-   小项目 / 先跑通：
    -   `AdamW + 固定 lr` 或 `AdamW + StepLR`
-   CV / NLP 中等规模模型：
    -   `SGD/AdamW + Warmup + CosineAnnealingLR`
-   想要自动根据 val_loss 调整：
    -   `任意 optimizer + ReduceLROnPlateau`
-   想折腾一点、追效果：
    -   `Warmup + CosineAnnealingWarmRestarts` 或 `OneCycleLR`

---

## 10. 一个典型 PyTorch 模板

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# 例：总共 100 epoch，使用余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6,
)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        loss = ...
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()  # 调整学习率
```
