# PyTorch: cat vs stack 张量拼接详解

## 1. 核心区别

### 一句话总结

- **torch.cat**: 在**已有维度**上拼接,不增加新维度
- **torch.stack**: 在**新维度**上堆叠,会增加一个维度

### 核心差异表

| 特性 | torch.cat | torch.stack |
|------|-----------|-------------|
| **维度变化** | 不变 | +1 |
| **输入要求** | 某些维度可以不同 | 所有维度必须完全相同 |
| **拼接方式** | 在指定维度上连接 | 创建新维度后堆叠 |
| **典型用途** | 合并不同长度的序列 | 批处理相同形状的数据 |

---

## 2. 直观理解

### torch.cat - 连接

想象把两根木棍**首尾相连**,拼成一根更长的木棍:

```python
import torch

# 两个 [3] 的tensor
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# cat: 首尾相连
result = torch.cat([a, b], dim=0)
print(result)
# tensor([1, 2, 3, 4, 5, 6])  ← 形状: [6]

print(f"维度变化: {a.dim()} → {result.dim()}")  # 1 → 1
```

### torch.stack - 堆叠

想象把两根木棍**平行堆叠**,像筷子一样:

```python
# 同样的两个tensor
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# stack: 平行堆叠
result = torch.stack([a, b], dim=0)
print(result)
# tensor([[1, 2, 3],
#         [4, 5, 6]])  ← 形状: [2, 3]

print(f"维度变化: {a.dim()} → {result.dim()}")  # 1 → 2
```

---

## 3. 维度详解

### torch.cat - 在已有维度拼接

```python
# 示例1: 在dim=0拼接
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(4, 3)  # [4, 3]  ← 第0维不同
c = torch.cat([a, b], dim=0)
print(c.shape)  # torch.Size([6, 3])  ← 2+4=6

# 示例2: 在dim=1拼接
a = torch.randn(2, 3)  # [2, 3]
b = torch.randn(2, 5)  # [2, 5]  ← 第1维不同
c = torch.cat([a, b], dim=1)
print(c.shape)  # torch.Size([2, 8])  ← 3+5=8
```

**关键规则**: 除了拼接维度外,其他维度必须相同

```python
# ❌ 错误示例
a = torch.randn(2, 3)
b = torch.randn(2, 4)
c = torch.cat([a, b], dim=0)  # RuntimeError!
# 因为dim=1不同(3 vs 4),无法在dim=0拼接
```

### torch.stack - 创建新维度

```python
# 所有tensor必须形状完全相同
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)

# 在dim=0堆叠
result = torch.stack([a, b, c], dim=0)
print(result.shape)  # torch.Size([3, 2, 3])

# 在dim=1堆叠
result = torch.stack([a, b, c], dim=1)
print(result.shape)  # torch.Size([2, 3, 3])

# 在dim=2堆叠
result = torch.stack([a, b, c], dim=2)
print(result.shape)  # torch.Size([2, 3, 3])
```

**关键规则**: 所有输入tensor形状必须完全相同

```python
# ❌ 错误示例
a = torch.randn(2, 3)
b = torch.randn(2, 4)  # 形状不同
c = torch.stack([a, b], dim=0)  # RuntimeError!
```

---

## 4. 图示对比

### cat: 在已有维度连接

```
dim=0方向cat:

a = [[1, 2, 3]]      [1, 2, 3]
    ↓           →   [4, 5, 6]
b = [[4, 5, 6]]     [7, 8, 9]

形状: [1, 3] + [2, 3] → [3, 3]


dim=1方向cat:

a = [[1, 2]]        [1, 2, 3, 4]
    ↓           →   [5, 6, 7, 8]
b = [[3, 4]]

形状: [2, 2] + [2, 2] → [2, 4]
```

### stack: 创建新维度堆叠

```
dim=0方向stack:

a = [1, 2, 3]       [[1, 2, 3],
    ↓           →    [4, 5, 6],
b = [4, 5, 6]        [7, 8, 9]]
    ↓
c = [7, 8, 9]

形状: 3个[3] → [3, 3]


dim=1方向stack:

a = [[1, 2],        [[[1, 2],
     [3, 4]]          [5, 6]],
    ↓           →
b = [[5, 6],         [[3, 4],
     [7, 8]]          [7, 8]]]

形状: 2个[2, 2] → [2, 2, 2]
```

---

## 5. 常见应用场景

### torch.cat 典型用例

#### 用例1: 拼接不同长度的序列

```python
# 文本处理: 拼接不同长度的句子
sentence1 = torch.randn(10, 512)  # 10个词
sentence2 = torch.randn(15, 512)  # 15个词
combined = torch.cat([sentence1, sentence2], dim=0)
print(combined.shape)  # torch.Size([25, 512])
```

#### 用例2: 特征拼接

```python
# 多模态学习: 拼接不同特征
image_features = torch.randn(batch, 2048)    # 图像特征
text_features = torch.randn(batch, 768)      # 文本特征
combined = torch.cat([image_features, text_features], dim=1)
print(combined.shape)  # torch.Size([batch, 2816])
```

#### 用例3: ResNet中的skip connection

```python
# 在通道维度拼接
def residual_block(x, residual):
    return torch.cat([x, residual], dim=1)  # 通道拼接

x = torch.randn(32, 64, 28, 28)
residual = torch.randn(32, 64, 28, 28)
output = residual_block(x, residual)
print(output.shape)  # torch.Size([32, 128, 28, 28])
```

### torch.stack 典型用例

#### 用例1: 构建batch

```python
# 从数据加载器收集单个样本
samples = []
for _ in range(batch_size):
    sample = torch.randn(3, 224, 224)  # 单张图片
    samples.append(sample)

batch = torch.stack(samples, dim=0)
print(batch.shape)  # torch.Size([batch_size, 3, 224, 224])
```

#### 用例2: 时间序列堆叠

```python
# RNN/LSTM: 堆叠不同时间步
timesteps = []
for t in range(seq_len):
    hidden = model(input_t)  # [batch, hidden_size]
    timesteps.append(hidden)

sequence = torch.stack(timesteps, dim=1)
print(sequence.shape)  # torch.Size([batch, seq_len, hidden_size])
```

#### 用例3: 多个预测结果集成

```python
# 模型集成: 堆叠多个模型的预测
predictions = []
for model in models:
    pred = model(x)  # [batch, num_classes]
    predictions.append(pred)

all_preds = torch.stack(predictions, dim=0)
# [num_models, batch, num_classes]

# 平均预测
final_pred = all_preds.mean(dim=0)
```

---

## 6. 相互转换

### stack可以用cat实现

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

# 方法1: 直接stack
result1 = torch.stack([a, b], dim=0)

# 方法2: unsqueeze + cat (等价)
result2 = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)

print(torch.equal(result1, result2))  # True
print(result1.shape)  # torch.Size([2, 2, 3])
```

### 理解它们的关系

```python
# stack本质上是:
# 1. 给每个tensor增加一个维度 (unsqueeze)
# 2. 在新维度上cat

def my_stack(tensors, dim=0):
    """手动实现stack"""
    expanded = [t.unsqueeze(dim) for t in tensors]
    return torch.cat(expanded, dim=dim)

a = torch.randn(2, 3)
b = torch.randn(2, 3)

result1 = torch.stack([a, b], dim=0)
result2 = my_stack([a, b], dim=0)

print(torch.equal(result1, result2))  # True
```

---

## 7. 性能对比

```python
import torch
import time

# 准备数据
tensors = [torch.randn(100, 100) for _ in range(100)]

# 测试1: stack性能
start = time.time()
for _ in range(1000):
    _ = torch.stack(tensors, dim=0)
print(f"stack: {time.time() - start:.4f}s")

# 测试2: unsqueeze + cat性能
start = time.time()
for _ in range(1000):
    expanded = [t.unsqueeze(0) for t in tensors]
    _ = torch.cat(expanded, dim=0)
print(f"unsqueeze + cat: {time.time() - start:.4f}s")

# 测试3: cat在已有维度
tensors_for_cat = [torch.randn(10, 100) for _ in range(100)]
start = time.time()
for _ in range(1000):
    _ = torch.cat(tensors_for_cat, dim=0)
print(f"cat: {time.time() - start:.4f}s")
```

**性能结论**:
- `stack` 比 `unsqueeze + cat` 更快(内部优化)
- `cat` 在已有维度上最快(不需要改变结构)

---

## 8. 常见陷阱

### 陷阱1: 忘记stack增加维度

```python
# ❌ 错误理解
tensors = [torch.randn(3, 4) for _ in range(5)]
result = torch.stack(tensors, dim=0)
print(result.shape)  # torch.Size([5, 3, 4])
# 很多人以为是 [3, 20] 或 [15, 4]
```

### 陷阱2: cat时维度不匹配

```python
# ❌ 错误
a = torch.randn(2, 3, 4)
b = torch.randn(2, 3, 5)
c = torch.cat([a, b], dim=0)  # RuntimeError!
# 应该用 dim=2
```

### 陷阱3: stack时形状不一致

```python
# ❌ 错误
a = torch.randn(2, 3)
b = torch.randn(2, 4)
c = torch.stack([a, b], dim=0)  # RuntimeError!
# 必须完全相同的形状
```

### 陷阱4: 空列表

```python
# ❌ 危险
tensors = []
result = torch.cat(tensors, dim=0)  # RuntimeError!

# ✅ 安全处理
if len(tensors) > 0:
    result = torch.cat(tensors, dim=0)
else:
    result = torch.empty(0)
```

---

## 9. 实用技巧

### 技巧1: 动态batch处理

```python
# DataLoader中的collate_fn
def collate_fn(batch):
    # batch是list of samples
    # 每个sample: (image, label)

    images = [item[0] for item in batch]  # 所有图片
    labels = [item[1] for item in batch]  # 所有标签

    # 用stack创建batch维度
    images_batch = torch.stack(images, dim=0)
    labels_batch = torch.stack(labels, dim=0)

    return images_batch, labels_batch
```

### 技巧2: 灵活的特征组合

```python
def combine_features(*features, method='cat'):
    """灵活组合特征"""
    if method == 'cat':
        # 在最后一维拼接
        return torch.cat(features, dim=-1)
    elif method == 'stack':
        # 创建新维度
        return torch.stack(features, dim=-1)
    elif method == 'sum':
        # 相加(要求形状相同)
        return sum(features)
    else:
        raise ValueError(f"Unknown method: {method}")

# 使用示例
f1 = torch.randn(32, 128)
f2 = torch.randn(32, 128)

result_cat = combine_features(f1, f2, method='cat')
print(result_cat.shape)  # torch.Size([32, 256])

result_stack = combine_features(f1, f2, method='stack')
print(result_stack.shape)  # torch.Size([32, 128, 2])
```

### 技巧3: 检查并自动选择

```python
def smart_combine(tensors, dim=0):
    """智能选择cat或stack"""
    if not tensors:
        raise ValueError("Empty tensor list")

    # 检查形状是否完全相同
    first_shape = tensors[0].shape
    all_same = all(t.shape == first_shape for t in tensors)

    if all_same:
        print("使用stack(形状相同)")
        return torch.stack(tensors, dim=dim)
    else:
        print("使用cat(形状不同)")
        return torch.cat(tensors, dim=dim)

# 示例
t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)
result1 = smart_combine([t1, t2], dim=0)  # 使用stack

t3 = torch.randn(4, 3)
result2 = smart_combine([t1, t3], dim=0)  # 使用cat
```

---

## 10. 快速参考

### 决策树

```
需要合并多个tensor?
│
├─ 想增加一个新维度? → torch.stack
│  └─ 要求: 所有tensor形状必须完全相同
│
└─ 在已有维度上拼接? → torch.cat
   └─ 要求: 除拼接维度外,其他维度相同
```

### API快查

```python
# torch.cat
torch.cat(
    tensors,      # list或tuple of tensors
    dim=0,        # 拼接的维度
    out=None      # 输出tensor(可选)
)

# torch.stack
torch.stack(
    tensors,      # list或tuple of tensors
    dim=0,        # 新维度的位置
    out=None      # 输出tensor(可选)
)
```

### 记忆口诀

> **cat是串联,stack是堆叠**
>
> - **cat**: 像链表一样**串**起来,维度不变
> - **stack**: 像书本一样**摞**起来,维度+1
>
> **选择原则**:
> - 需要批处理 → stack
> - 需要序列拼接 → cat
> - 形状必须相同 → stack
> - 某维可以不同 → cat

---

## 11. 实战案例

### 案例1: Transformer中的多头注意力

```python
def multi_head_attention(Q, K, V, num_heads=8):
    """
    Q, K, V: [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # 分割成多个head
    Q = Q.view(batch_size, seq_len, num_heads, d_k)
    K = K.view(batch_size, seq_len, num_heads, d_k)
    V = V.view(batch_size, seq_len, num_heads, d_k)

    # 每个head计算attention
    heads = []
    for i in range(num_heads):
        q = Q[:, :, i, :]
        k = K[:, :, i, :]
        v = V[:, :, i, :]
        head = compute_attention(q, k, v)
        heads.append(head)

    # 使用cat合并所有head
    output = torch.cat(heads, dim=-1)
    return output
```

### 案例2: GAN中的批量生成

```python
def generate_batch(generator, noise_dim=100, batch_size=32):
    """生成一批假样本"""
    # 生成多个噪声向量
    noises = []
    for _ in range(batch_size):
        noise = torch.randn(noise_dim)
        noises.append(noise)

    # 用stack组成batch
    noise_batch = torch.stack(noises, dim=0)

    # 生成
    fake_images = generator(noise_batch)
    return fake_images
```

### 案例3: 数据增强

```python
def augment_and_stack(image, num_augments=5):
    """
    对一张图片进行多次增强并堆叠
    image: [C, H, W]
    返回: [num_augments, C, H, W]
    """
    augmented = []
    for _ in range(num_augments):
        aug_img = random_augment(image)  # 随机增强
        augmented.append(aug_img)

    # stack创建batch维度
    return torch.stack(augmented, dim=0)

# 使用
image = torch.randn(3, 224, 224)
batch = augment_and_stack(image, num_augments=8)
print(batch.shape)  # torch.Size([8, 3, 224, 224])
```

---

## 12. 总结

### 核心要点

1. **维度变化**
   - `cat`: 维度数不变,指定维度的大小相加
   - `stack`: 维度数+1,在新维度上堆叠

2. **输入要求**
   - `cat`: 除了拼接维度,其他维度必须相同
   - `stack`: 所有维度必须完全相同

3. **使用场景**
   - `cat`: 序列拼接、特征融合、不同长度数据合并
   - `stack`: 批处理、时间序列、模型集成

4. **等价关系**
   ```python
   torch.stack(tensors, dim=d)
   ==
   torch.cat([t.unsqueeze(d) for t in tensors], dim=d)
   ```

### 最后建议

✅ **使用cat当**:
- 需要拼接变长序列
- 合并不同维度的特征
- 在已有维度上扩展

✅ **使用stack当**:
- 需要构建batch
- 收集时间序列
- 所有tensor形状相同且需要新维度

记住:**cat串联,stack堆叠**,根据实际需求选择合适的操作!
