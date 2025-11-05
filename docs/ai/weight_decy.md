# 权重衰减(Weight Decay)简介

## 什么是权重衰减

权重衰减是深度学习中一种重要的正则化技术，通过在训练过程中逐渐减小模型权重的大小，防止模型过拟合，提高泛化能力。它相当于在损失函数中添加了一个惩罚项，使得权重不会过大。

简而言之，权重衰退就是让权重 w 往 0 的方向拉，从而保证其上下一致，从而表现稳定，因为如果 w 参数各异，实际上模型是不稳定的，也就是曲线是不平滑的。

## 数学原理

在标准的损失函数基础上，权重衰减添加了一个正则化项：

$$
L = L_0 + \frac{\lambda}{2} \times ||w||^2
$$

其中：
- **L₀** 是原始损失函数（如交叉熵损失）
- **λ** 是权重衰减系数，控制正则化强度
- **||w||²** 是所有权重参数的L2范数平方

在梯度下降更新时，权重更新公式变为：

$$
w \leftarrow w - \alpha \times \left(\frac{\partial L_0}{\partial w} + \lambda w\right) = (1 - \alpha\lambda)w - \alpha \times \frac{\partial L_0}{\partial w}
$$

可以看到，每次更新时权重都会乘以一个小于1的因子 **(1 - αλ)**，这就是"衰减"的由来。

## 实现方式

### 1. L2正则化方式

直接在损失函数中添加权重的L2范数惩罚项，让优化器在计算梯度时自然包含正则化项。

### 2. 显式权重衰减

在每次权重更新时，显式地对权重进行衰减。例如PyTorch的AdamW优化器就采用这种方式，与标准Adam在自适应学习率场景下效果更好。

## 在NLP中的应用

权重衰减在NLP领域应用广泛：

- BERT、GPT等预训练模型的微调通常使用**0.01**的权重衰减系数
- Transformer模型训练时，权重衰减是标准配置
- 对**LayerNorm层**和**Embedding层**通常不应用权重衰减
- **偏置项(bias)** 一般也不使用权重衰减

## 实践建议

### 选择合适的权重衰减系数

- 常用范围：**0.01 ~ 0.1**
- 数据量小时可以增大系数，防止过拟合
- 数据量大时可以减小系数

### 注意事项

- 需要与学习率协同调优
- 使用AdamW时，建议采用**解耦的权重衰减**
- 监控训练和验证损失，调整系数避免欠拟合或过拟合

## 代码示例

### PyTorch中的使用

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Linear(10, 1)

# 方式1: 使用Adam + L2正则化（不推荐）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# 方式2: 使用AdamW（推荐）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 方式3: 对不同参数组使用不同的权重衰减
optimizer = optim.AdamW([
    {'params': model.weight, 'weight_decay': 0.01},
    {'params': model.bias, 'weight_decay': 0.0}  # bias不使用权重衰减
], lr=0.001)
```

### Hugging Face Transformers中的使用

```python
from transformers import AdamW, get_linear_schedule_with_warmup

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

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
```

## 权重衰减 vs Dropout

| 特性 | 权重衰减 | Dropout |
|------|---------|---------|
| 作用时机 | 训练时每次权重更新 | 训练时随机关闭神经元 |
| 推理阶段 | 无影响 | 需要关闭 |
| 计算开销 | 很小 | 中等 |
| 适用场景 | 几乎所有模型 | 主要用于全连接层 |
| NLP中常用性 | 必备 | 较少使用 |

## 总结

权重衰减是深度学习中简单但有效的正则化方法，在NLP任务中几乎是必不可少的技术。通过限制模型权重的大小，它能够有效防止过拟合，提高模型的泛化能力。

**关键要点：**
- 权重衰减通过L2正则化约束模型权重
- 在NLP中广泛应用，常用系数为0.01
- 使用AdamW时采用解耦的权重衰减效果更好
- LayerNorm、Embedding和bias通常不使用权重衰减
- 需要与学习率等超参数协同调优