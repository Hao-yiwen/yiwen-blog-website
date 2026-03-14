# PyTorch: cat vs stack 张量拼接

## 核心区别

| 特性 | torch.cat | torch.stack |
|------|-----------|-------------|
| **维度变化** | 不变 | +1 |
| **输入要求** | 某些维度可不同 | 所有维度必须相同 |
| **拼接方式** | 在已有维度上连接 | 创建新维度后堆叠 |
| **典型用途** | 序列拼接、特征融合 | 批处理、时间序列 |

**一句话总结**:
- `cat`: 在已有维度上**串联**,像首尾相连
- `stack`: 创建新维度**堆叠**,像书本摞起来

---

## 基础示例

### torch.cat - 连接

```python
import torch

# 在dim=0拼接
a = torch.tensor([1, 2, 3])  # [3]
b = torch.tensor([4, 5, 6])  # [3]
result = torch.cat([a, b], dim=0)
print(result)  # tensor([1, 2, 3, 4, 5, 6])  形状: [6]

# 二维tensor拼接
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(4, 3)  # [4, 3]  ← 第0维不同
c = torch.cat([a, b], dim=0)
print(c.shape)  # torch.Size([6, 3])  ← 2+4=6
```

**关键规则**: 除拼接维度外,其他维度必须相同

### torch.stack - 堆叠

```python
# 一维tensor堆叠
a = torch.tensor([1, 2, 3])  # [3]
b = torch.tensor([4, 5, 6])  # [3]
result = torch.stack([a, b], dim=0)
print(result)
# tensor([[1, 2, 3],
#         [4, 5, 6]])  形状: [2, 3]

# 二维tensor堆叠
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(2, 3)  # [2, 3]  ← 形状必须完全相同
c = torch.stack([a, b], dim=0)
print(c.shape)  # torch.Size([2, 2, 3])  ← 新增了一个维度
```

**关键规则**: 所有输入tensor形状必须完全相同

---

## 常见应用场景

### torch.cat 用例

```python
# 1. 特征拼接
image_features = torch.randn(32, 2048)  # 图像特征
text_features = torch.randn(32, 768)     # 文本特征
combined = torch.cat([image_features, text_features], dim=1)
print(combined.shape)  # torch.Size([32, 2816])

# 2. 序列拼接
sentence1 = torch.randn(10, 512)  # 10个词
sentence2 = torch.randn(15, 512)  # 15个词
combined = torch.cat([sentence1, sentence2], dim=0)
print(combined.shape)  # torch.Size([25, 512])
```

### torch.stack 用例

```python
# 1. 构建batch
samples = [torch.randn(3, 224, 224) for _ in range(32)]
batch = torch.stack(samples, dim=0)
print(batch.shape)  # torch.Size([32, 3, 224, 224])

# 2. 时间序列堆叠 (RNN/LSTM)
timesteps = [torch.randn(32, 128) for _ in range(10)]  # 10个时间步
sequence = torch.stack(timesteps, dim=1)
print(sequence.shape)  # torch.Size([32, 10, 128])

# 3. 模型集成
predictions = [model(x) for model in models]  # 多个模型预测
all_preds = torch.stack(predictions, dim=0)
final_pred = all_preds.mean(dim=0)  # 平均预测
```

---

## 相互转换

`stack` 本质上是 `unsqueeze + cat`:

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# 这两种方式等价
result1 = torch.stack([a, b], dim=0)
result2 = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)

print(torch.equal(result1, result2))  # True
```

---

## 常见陷阱

```python
# ❌ 陷阱1: cat时维度不匹配
a = torch.randn(2, 3)
b = torch.randn(2, 4)
c = torch.cat([a, b], dim=0)  # RuntimeError! dim=1不同(3 vs 4)

# ❌ 陷阱2: stack时形状不一致
a = torch.randn(2, 3)
b = torch.randn(2, 4)
c = torch.stack([a, b], dim=0)  # RuntimeError! 必须完全相同

# ❌ 陷阱3: 忘记stack会增加维度
tensors = [torch.randn(3, 4) for _ in range(5)]
result = torch.stack(tensors, dim=0)
print(result.shape)  # torch.Size([5, 3, 4]) 不是 [15, 4]!
```

---

## 快速参考

### 决策树

```
需要合并多个tensor?
├─ 想增加新维度 → torch.stack (要求形状完全相同)
└─ 在已有维度拼接 → torch.cat (拼接维度可不同)
```

### API

```python
torch.cat(tensors, dim=0)    # 在dim维度连接
torch.stack(tensors, dim=0)  # 在dim位置创建新维度
```

### 记忆口诀

> **cat串联,stack堆叠**
> - cat: 像链表串起来,维度不变
> - stack: 像书本摞起来,维度+1

**选择原则**:
- 批处理、形状相同 → `stack`
- 序列拼接、某维不同 → `cat`
