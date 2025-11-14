---
title: Adam vs SGD 优化器深度对比
sidebar_label: Adam vs SGD 优化器
date: 2025-11-14
last_update:
  date: 2025-11-14
---

# Adam vs SGD 优化器深度对比

## 核心区别概览

在深度学习训练中，**SGD（随机梯度下降）** 和 **Adam（自适应矩估计）** 是两种最常用的优化器，它们代表了不同的优化哲学：

**SGD = 稳健的基础优化器**（简单直接，需要精细调参）
**Adam = 自适应优化器**（智能灵活,对超参数更宽容）

---

## 快速对比表

| 特性 | SGD | Adam |
|------|-----|------|
| **参数更新方式** | 仅基于当前梯度 | 基于梯度的一阶矩和二阶矩估计 |
| **学习率敏感度** | 非常敏感，需要精细调整 | 相对不敏感，自适应调整 |
| **收敛速度** | 较慢，需要更多迭代 | 快速，尤其在训练初期 |
| **最终性能** | 通常更好，泛化能力强 | 可能略差，容易陷入尖锐最小值 |
| **适用场景** | CNN、图像分类、ResNet | Transformer、NLP、生成模型 |
| **内存开销** | 低（仅存储梯度） | 高（需存储一阶和二阶矩） |
| **超参数调优** | 复杂（lr、momentum、weight_decay） | 简单（通常使用默认值即可） |

---

## 数学原理

### SGD：基于当前梯度的直接更新

SGD 的更新规则非常直接：

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
$$

其中：
- $\theta$ 是模型参数
- $\eta$ 是学习率
- $\nabla_\theta L$ 是损失函数的梯度

**带动量的 SGD (Momentum)**：

$$
\begin{align}
v_{t+1} &= \mu \cdot v_t + \nabla_\theta L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \cdot v_{t+1}
\end{align}
$$

其中 $\mu$ 是动量系数（通常为 0.9），$v$ 是速度向量。

**直觉理解**：
- 梯度大 → 步长大
- 梯度小 → 步长小
- 所有参数使用相同的学习率
- Momentum 增加了"惯性"，帮助跨越局部最优

---

### Adam：自适应学习率优化

Adam 结合了 RMSprop 和 Momentum 的优点，为每个参数自适应地调整学习率。

**核心思想**：
1. **一阶矩估计（$m_t$）**：梯度的指数移动平均（方向）
2. **二阶矩估计（$v_t$）**：梯度平方的指数移动平均（尺度）

**更新规则**：

$$
\begin{align}
m_t &= \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot \nabla_\theta L(\theta_t) \\
v_t &= \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot \nabla_\theta L(\theta_t)^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \quad \text{(偏差修正)} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
\end{align}
$$

默认超参数：
- $\beta_1 = 0.9$（一阶矩衰减率）
- $\beta_2 = 0.999$（二阶矩衰减率）
- $\epsilon = 10^{-8}$（数值稳定性常数）

**直觉理解**：
- $m_t$：判断应该往哪个方向走
- $v_t$：判断每个参数的步长应该多大
- $\sqrt{\hat{v}_t}$：自动归一化学习率，梯度变化大的参数步长变小
- 每个参数都有自己的"有效学习率"

---

## 训练表现对比

### 收敛速度

```python
# 典型的训练损失曲线特征

# SGD：
# - 初期下降较慢
# - 中后期稳定下降
# - 可能出现震荡
# - 需要精心设计学习率调度

# Adam：
# - 初期快速下降
# - 几个 epoch 内损失显著降低
# - 后期可能进入平台期
# - 对学习率调度不敏感
```

### 泛化性能

**为什么 SGD 通常泛化更好？**

1. **平坦最小值 vs 尖锐最小值**
   - SGD 更容易找到"平坦"的最小值（损失面较宽）
   - Adam 可能陷入"尖锐"的最小值（损失面较窄）
   - 平坦最小值对参数扰动不敏感 → 更好的泛化能力

2. **噪声的作用**
   - SGD 的梯度噪声帮助探索参数空间
   - Adam 的自适应机制可能过早收敛

3. **实践证据**
   - ImageNet 分类任务：SGD 通常比 Adam 高 1-2%
   - CIFAR-10/100：SGD 训练的模型测试准确率更高

---

## 使用场景指南

### 优先使用 Adam 的场景

✅ **Transformer 模型**
- BERT、GPT、T5 等所有预训练语言模型
- Vision Transformer (ViT)
- 实践中通常使用 **AdamW**（带解耦权重衰减的 Adam）

✅ **生成模型**
- Diffusion Models (Stable Diffusion)
- GAN（生成对抗网络）
- VAE（变分自编码器）

✅ **强化学习**
- 策略梯度方法
- Actor-Critic 算法

✅ **快速原型开发**
- 需要快速验证想法
- 不想花时间调超参数

✅ **稀疏梯度场景**
- NLP 任务（词嵌入更新稀疏）
- 推荐系统

**代码示例**：

```python
# Transformer 训练标准配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 学习率预热 + 余弦退火
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=100000
)
```

---

### 优先使用 SGD 的场景

✅ **图像分类任务**
- ResNet、VGG、MobileNet 在 ImageNet 上训练
- CIFAR-10/100 分类
- 医学图像分类

✅ **追求最佳泛化性能**
- 对准确率有极致要求
- 测试集性能比训练速度更重要

✅ **已有成熟的超参数配置**
- 复现经典论文
- 使用现有最佳实践

✅ **计算资源有限**
- Adam 需要 2 倍的参数内存（m 和 v）
- SGD 内存开销更小

**代码示例**：

```python
# ResNet 训练标准配置
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True  # Nesterov 动量通常更好
)

# 多步学习率衰减
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # 在第30、60、90个epoch降低学习率
    gamma=0.1  # 每次乘以0.1
)
```

---

## 高级技巧与最佳实践

### 混合策略：Adam → SGD

许多顶级竞赛和研究团队使用混合策略：

```python
# 训练分两阶段
# 阶段1（前70%训练）：Adam 快速收敛
optimizer_adam = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(70):
    train_one_epoch(model, optimizer_adam)

# 阶段2（后30%训练）：切换到 SGD 提升泛化
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=1e-4
)

for epoch in range(30):
    train_one_epoch(model, optimizer_sgd)
```

### 学习率调度策略

**SGD 必须配合学习率调度器**，否则效果通常不如 Adam：

```python
# 常见的调度器选择

# 1. StepLR：固定间隔降低学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 2. MultiStepLR：在指定epoch降低学习率
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)

# 3. CosineAnnealingLR：余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 4. ReduceLROnPlateau：验证集性能停止提升时降低学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

### 学习率预热（Warmup）

对于 Adam 和大模型训练，学习率预热非常重要：

```python
def get_lr_with_warmup(optimizer, step, warmup_steps, max_lr):
    """线性预热 + 余弦衰减"""
    if step < warmup_steps:
        # 预热阶段：线性增加
        lr = max_lr * step / warmup_steps
    else:
        # 训练阶段：余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = max_lr * 0.5 * (1 + np.cos(np.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

---

## Adam vs AdamW：重要区别

**AdamW** 是 Adam 的改进版本，修复了权重衰减的实现问题。

### 问题：Adam 中的 L2 正则化 bug

在标准 Adam 中，`weight_decay` 参数实际上是将 L2 正则化项加到梯度中：

$$
\nabla_\theta L = \nabla_\theta L_0 + \lambda \theta
$$

这导致 L2 正则化被 Adam 的自适应学习率机制"稀释"了，效果不如预期。

### 解决方案：AdamW 的解耦权重衰减

AdamW 将权重衰减与梯度更新解耦：

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

**实践建议**：
- ✅ 使用 `torch.optim.AdamW` 而不是 `torch.optim.Adam`
- ✅ 几乎所有现代 Transformer 模型都使用 AdamW
- ✅ 典型的 `weight_decay` 值：0.01 ~ 0.1

```python
# 推荐：使用 AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # 正确的权重衰减
)

# 不推荐：使用 Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # 实际效果不符合预期
)
```

---

## 超参数调优建议

### SGD 超参数

| 参数 | 典型范围 | 调优建议 |
|------|---------|---------|
| `lr` | 0.001 ~ 0.1 | 最重要的超参数，建议网格搜索 |
| `momentum` | 0.9 ~ 0.99 | 通常使用 0.9，大模型可用 0.99 |
| `weight_decay` | 1e-5 ~ 1e-3 | 与 lr 协同调整 |
| `nesterov` | True/False | 通常设为 True |

**学习率选择经验**：
- 小模型/小数据集：0.01 ~ 0.1
- 中等规模：0.001 ~ 0.01
- 大模型：0.0001 ~ 0.001

---

### Adam/AdamW 超参数

| 参数 | 典型值 | 调优建议 |
|------|--------|---------|
| `lr` | 1e-4 ~ 1e-3 | 通常从 1e-3 开始 |
| `betas` | (0.9, 0.999) | 很少需要改变 |
| `eps` | 1e-8 | 保持默认即可 |
| `weight_decay` | 0.01 ~ 0.1 | Transformer 常用 0.01 |

**学习率选择经验**：
- NLP 预训练：1e-4 ~ 5e-5
- NLP 微调：1e-5 ~ 5e-5
- 视觉任务：1e-3 ~ 1e-4
- 强化学习：3e-4

---

## 完整训练示例对比

### 使用 SGD 训练 ResNet

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

# 模型和优化器
model = resnet50(pretrained=False, num_classes=10)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# 学习率调度
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1
)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 每个epoch结束后更新学习率
    scheduler.step()

    # 验证
    model.eval()
    # ... 验证代码 ...
```

---

### 使用 AdamW 训练 Transformer

```python
import torch
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup

# 模型
model = BertModel.from_pretrained('bert-base-uncased')

# 区分需要和不需要权重衰减的参数
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]

# 优化器
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度（带预热）
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = num_training_steps // 10  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # 梯度裁剪（Transformer 训练常用）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 每个batch后更新学习率
```

---

## 常见问题与解决方案

### Q1: 为什么我的 SGD 训练效果很差？

可能原因：
1. **学习率太大或太小**
   - 尝试网格搜索：`[0.1, 0.01, 0.001]`
   - 观察训练初期损失变化

2. **没有使用 momentum**
   - 至少设置 `momentum=0.9`

3. **没有学习率调度**
   - SGD 几乎总是需要学习率衰减

4. **初始化不当**
   - 确保使用合适的权重初始化（如 He/Xavier）

---

### Q2: Adam 为什么会过拟合？

Adam 的自适应学习率可能导致：
- 快速进入尖锐最小值
- 对训练集过度拟合

**解决方案**：
- ✅ 使用 AdamW 而不是 Adam
- ✅ 增大 `weight_decay`（如 0.1）
- ✅ 添加 Dropout 或其他正则化
- ✅ 使用学习率预热和衰减
- ✅ 考虑后期切换到 SGD

---

### Q3: 什么时候应该切换优化器？

**从 Adam 切换到 SGD**：
- 训练后期（70-80% 训练完成）
- 验证集性能停止提升
- 需要榨取最后的性能提升

**切换步骤**：
```python
# 保存 Adam 训练的模型
torch.save(model.state_dict(), 'model_adam.pth')

# 创建新的 SGD 优化器（使用较小的学习率）
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,  # 比 Adam 的 lr 小
    momentum=0.9,
    weight_decay=1e-4
)

# 继续训练
for epoch in range(remaining_epochs):
    # ...
```

---

### Q4: 如何选择 batch size？

Batch size 会影响优化器的行为：

**大 batch size（256+）**：
- SGD：可能需要线性缩放学习率（`lr = base_lr * batch_size / 256`）
- Adam：相对稳定，但可能需要更长的预热

**小 batch size（<32）**：
- SGD：梯度噪声大，可能需要降低学习率
- Adam：通常表现更稳定

**实践建议**：
- 尽可能使用大 batch size（受显存限制）
- 使用梯度累积模拟大 batch
- 调整学习率与 batch size 成正比

---

## 性能对比实验

以下是在常见任务上的典型性能对比：

### ImageNet 分类（ResNet-50）

| 优化器 | Top-1 准确率 | 训练时间 | 最佳学习率 |
|--------|-------------|---------|-----------|
| SGD + Momentum | **76.5%** | 基准 | 0.1 |
| Adam | 74.8% | 0.9x | 1e-3 |
| AdamW | 75.2% | 0.9x | 1e-3 |

---

### BERT 预训练（BookCorpus + Wikipedia）

| 优化器 | 验证损失 | 训练时间 | 最佳学习率 |
|--------|---------|---------|-----------|
| SGD + Momentum | 3.2 | 1.5x | 1e-3 |
| Adam | 2.9 | 基准 | 1e-4 |
| AdamW | **2.7** | 基准 | 1e-4 |

---

## 总结与建议

### 快速决策树

```
你的任务是什么？
│
├─ Transformer / NLP 任务？
│  └─ ✅ 使用 AdamW (lr=1e-4 to 1e-3, weight_decay=0.01)
│
├─ CNN / 图像分类？
│  └─ ✅ 使用 SGD (lr=0.1, momentum=0.9, 多步衰减)
│
├─ 生成模型 (GAN, Diffusion)？
│  └─ ✅ 使用 Adam/AdamW (lr=1e-4)
│
├─ 强化学习？
│  └─ ✅ 使用 Adam (lr=3e-4)
│
└─ 不确定？
   └─ ✅ 先用 Adam 快速验证，再用 SGD 调优
```

---

### 核心要点

1. **Adam 系列**（Adam/AdamW）
   - ✅ 训练速度快，对超参数宽容
   - ✅ 适合 NLP、生成模型、强化学习
   - ✅ 现代深度学习的首选（尤其是 Transformer）
   - ❌ 可能泛化性能略差（图像任务）
   - ❌ 内存开销更大

2. **SGD + Momentum**
   - ✅ 最终性能通常最好（图像分类）
   - ✅ 内存开销小
   - ✅ 理论基础扎实，行为可预测
   - ❌ 需要精细调参（lr, momentum, weight_decay）
   - ❌ 训练初期收敛慢
   - ❌ 必须配合学习率调度

3. **实践建议**
   - 使用 **AdamW** 而不是 Adam（几乎总是更好）
   - SGD 必须使用 **momentum** 和 **学习率调度**
   - 大模型训练使用 **学习率预热**
   - 图像任务考虑 **Adam → SGD** 两阶段训练
   - 始终监控 **训练/验证曲线** 调整策略

---

## 参考文献

1. **SGD 原始论文**
   Robbins, H., & Monro, S. (1951). *A stochastic approximation method*

2. **Momentum**
   Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods*

3. **Adam**
   Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*

4. **AdamW**
   Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization*

5. **泛化性能分析**
   Wilson, A. C., et al. (2017). *The marginal value of adaptive gradient methods in machine learning*

6. **平坦最小值理论**
   Keskar, N. S., et al. (2017). *On large-batch training for deep learning: Generalization gap and sharp minima*
