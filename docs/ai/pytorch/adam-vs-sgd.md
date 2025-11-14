---
title: Adam vs SGD 优化器对比
sidebar_label: Adam vs SGD 优化器
date: 2025-11-14
last_update:
  date: 2025-11-14
---

# Adam vs SGD 优化器对比

## 一句话总结

**SGD = 稳健但慢**（需要精细调参，泛化能力强）
**Adam = 快速但需注意**（自适应学习率，对超参数宽容）

---

## 核心区别

| 特性 | SGD | Adam |
|------|-----|------|
| **更新方式** | 仅基于当前梯度 | 基于梯度的一阶矩和二阶矩 |
| **学习率** | 非常敏感 | 相对不敏感，自适应调整 |
| **收敛速度** | 慢 | 快 |
| **最终性能** | 通常更好，泛化强 | 可能略差 |
| **适用场景** | CNN、图像分类 | Transformer、NLP、生成模型 |
| **内存开销** | 低 | 高（2倍参数内存） |
| **调参难度** | 难 | 简单 |

---

## 数学原理简述

### SGD + Momentum

$$
\begin{align}
v_{t+1} &= \mu \cdot v_t + \nabla_\theta L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \cdot v_{t+1}
\end{align}
$$

- $\mu$：动量系数（通常 0.9）
- $\eta$：学习率
- 所有参数使用相同学习率

### Adam

$$
\begin{align}
m_t &= \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot \nabla_\theta L \\
v_t &= \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot \nabla_\theta L^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t
\end{align}
$$

- $m_t$：梯度的指数移动平均（方向）
- $v_t$：梯度平方的指数移动平均（尺度）
- 每个参数有自己的"有效学习率"

---

## 使用场景

### 优先用 Adam/AdamW

- ✅ **Transformer**（BERT、GPT、ViT）
- ✅ **生成模型**（Diffusion、GAN、VAE）
- ✅ **强化学习**
- ✅ **快速原型开发**
- ✅ **NLP 任务**（稀疏梯度）

```python
# Transformer 标准配置
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

### 优先用 SGD

- ✅ **图像分类**（ResNet、VGG、MobileNet）
- ✅ **追求最佳泛化性能**
- ✅ **计算资源有限**（内存小）
- ✅ **复现经典论文**

```python
# ResNet 标准配置
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

# 必须配合学习率调度
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1
)
```

---

## Adam vs AdamW：重要！

**问题**：标准 Adam 的 `weight_decay` 实现有 bug，L2 正则化被自适应学习率稀释。

**解决**：AdamW 将权重衰减与梯度更新解耦。

```python
# ✅ 推荐：AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # 正确的权重衰减
)

# ❌ 不推荐：Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # 实际效果不符合预期
)
```

**结论**：几乎总是用 **AdamW** 而不是 Adam！

---

## 高级技巧

### 混合策略：Adam → SGD

```python
# 阶段1：Adam 快速收敛（前70%）
optimizer_adam = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(70):
    train_one_epoch(model, optimizer_adam)

# 阶段2：SGD 提升泛化（后30%）
optimizer_sgd = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=1e-4
)
for epoch in range(30):
    train_one_epoch(model, optimizer_sgd)
```

### 学习率预热（大模型必备）

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,  # 预热步数
    num_training_steps=100000
)
```

---

## 超参数建议

### SGD

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `lr` | 0.001 ~ 0.1 | 最重要，需网格搜索 |
| `momentum` | 0.9 | 几乎总是需要 |
| `weight_decay` | 1e-5 ~ 1e-3 | 与 lr 协同调整 |
| `nesterov` | True | 建议开启 |

### Adam/AdamW

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `lr` | 1e-4 ~ 1e-3 | 从 1e-3 开始 |
| `betas` | (0.9, 0.999) | 保持默认 |
| `weight_decay` | 0.01 ~ 0.1 | Transformer 常用 0.01 |

---

## 常见问题

### Q: SGD 效果很差？

可能原因：
1. 学习率不合适（试试 `[0.1, 0.01, 0.001]`）
2. 没用 momentum（至少 0.9）
3. 没用学习率调度（SGD 必须配合）

### Q: Adam 过拟合？

解决方案：
- 用 AdamW 而不是 Adam
- 增大 `weight_decay`
- 考虑后期切换到 SGD

### Q: 如何选择？

```
任务类型？
├─ Transformer/NLP → AdamW
├─ CNN/图像分类 → SGD
├─ 生成模型 → Adam/AdamW
├─ 强化学习 → Adam
└─ 不确定 → 先 Adam 验证，再 SGD 调优
```

---

## 性能对比

### ImageNet（ResNet-50）

| 优化器 | Top-1 准确率 |
|--------|-------------|
| SGD + Momentum | **76.5%** |
| AdamW | 75.2% |

### BERT 预训练

| 优化器 | 验证损失 |
|--------|---------|
| AdamW | **2.7** |
| SGD + Momentum | 3.2 |

---

## 总结

### 核心要点

1. **AdamW**
   - ✅ 快速、方便、适合 NLP 和生成模型
   - ✅ 对超参数不敏感
   - ❌ 图像分类可能泛化略差

2. **SGD + Momentum**
   - ✅ 最终性能最好（图像任务）
   - ✅ 内存小、行为可预测
   - ❌ 必须精细调参和学习率调度

3. **实践建议**
   - 优先用 **AdamW**（不是 Adam）
   - 图像任务考虑 SGD 或混合策略
   - 大模型必须用学习率预热
   - 始终监控训练/验证曲线
