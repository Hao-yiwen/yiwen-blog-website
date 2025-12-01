# PyTorch: view vs reshape 与连续性

## 1. 什么是连续性（Contiguous）

### 核心概念

**连续性 = tensor在内存中的存储顺序和逻辑顺序一致**

### 直观理解

```python
import torch

# 创建一个 2x3 的tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("逻辑视图（你看到的）:")
# [[1, 2, 3],
#  [4, 5, 6]]

print("物理内存（实际存储）:")
# [1, 2, 3, 4, 5, 6]  ← 连续存储，一个接一个
```

这就是**连续的**：从左到右、从上到下读取时，内存中的数据也是这个顺序。

### 转置后变成不连续

```python
y = x.t()  # 转置

print("逻辑视图:")
# [[1, 4],
#  [2, 5],
#  [3, 6]]

print("物理内存（未改变！）:")
# [1, 2, 3, 4, 5, 6]  ← 还是原来的顺序

print(y.is_contiguous())  # False
```

**关键点**：转置后,PyTorch并没有重新排列内存中的数据，而是通过改变**stride（步长）**来改变访问方式。

### Stride（步长）详解

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(x.stride())  # (3, 1)
# 含义：
# - 行方向移动1步 → 内存地址 +3
# - 列方向移动1步 → 内存地址 +1

y = x.t()
print(y.stride())  # (1, 3)
# 含义：
# - 行方向移动1步 → 内存地址 +1
# - 列方向移动1步 → 内存地址 +3
```

### 图示

```
原始 x (连续):
逻辑:        内存:
[1 2 3]     [1][2][3][4][5][6]
[4 5 6]      ↓  ↓  ↓  ↓  ↓  ↓
            顺序访问

转置 y (不连续):
逻辑:        内存:
[1 4]       [1][2][3][4][5][6]
[2 5]        ↓     ↓     (跳着访问)
[3 6]           ↓     ↓
                   ↓     ↓
```

### 哪些操作会导致不连续

```python
x = torch.randn(2, 3, 4)

# 导致不连续：
y1 = x.transpose(0, 1)     # 转置
y2 = x.permute(2, 0, 1)    # 重排维度
y3 = x[:, :, ::2]          # 跳步切片
y4 = x.narrow(1, 0, 2)     # narrow操作

# 保持连续：
z1 = x + 1                 # 数学运算（创建新tensor）
z2 = x.clone()             # 克隆
z3 = x.reshape(...)        # reshape会自动处理
```

---

## 2. view vs reshape

### 核心区别

| 特性 | view | reshape |
|------|------|---------|
| **连续性要求** | 必须连续，否则报错 | 自动处理，不连续时会复制 |
| **返回值** | 总是返回视图（共享内存） | 可能返回视图或副本 |
| **速度** | 快（不复制数据） | 不连续时较慢（需复制） |
| **安全性** | 严格，问题立即暴露 | 宽松，可能隐藏性能问题 |

### 代码示例

#### 示例1：连续tensor

```python
x = torch.randn(2, 3)

# 两者都可以工作，且都返回视图
y1 = x.view(6)
y2 = x.reshape(6)

# 都共享内存
y1[0] = 999
print(x[0, 0])  # 999

y2[0] = 888
print(x[0, 0])  # 888
```

#### 示例2：不连续tensor

```python
x = torch.randn(2, 3)
y = x.t()  # 转置，不连续

# view: 报错
try:
    z = y.view(6)
except RuntimeError as e:
    print("view报错:", e)

# reshape: 自动复制数据，成功
z = y.reshape(6)  # OK，但复制了数据
print(z.shape)  # torch.Size([6])
```

#### 示例3：检查是否共享内存

```python
x = torch.randn(2, 3)

# 连续情况
y = x.reshape(6)
print(y.data_ptr() == x.data_ptr())  # True，共享内存

# 不连续情况
x_t = x.t()
z = x_t.reshape(6)
print(z.data_ptr() == x_t.data_ptr())  # False，复制了数据
```

### view的典型用法

```python
# 明确知道tensor是连续的
batch_size = 32
x = torch.randn(batch_size, 3, 224, 224)  # 刚创建，必然连续
x = x.view(batch_size, -1)  # 展平

# 保证共享内存
x = torch.randn(10, 20)
y = x.view(-1)  # 如果不连续会报错，提醒你
y[0] = 999  # 100%会影响x
```

### reshape的典型用法

```python
# 不确定tensor是否连续
def process_input(x):
    # x可能经过了各种操作，不确定连续性
    return x.reshape(batch_size, -1)  # 安全

# 快速原型开发
x = some_complex_operations(data)
x = x.reshape(new_shape)  # 不用担心连续性
```

### 如何处理不连续

```python
x = torch.randn(2, 3).t()  # 不连续

# 方法1: 使用 reshape（自动处理）
y = x.reshape(-1)

# 方法2: 显式转为连续后用 view（推荐）
y = x.contiguous().view(-1)  # 更清晰地表达意图

# 方法3: 检查后决定
if x.is_contiguous():
    y = x.view(-1)
else:
    y = x.contiguous().view(-1)
```

---

## 3. 实用建议

### 什么时候用 view

✅ **明确知道tensor是连续的**
```python
x = torch.randn(batch, channels, h, w)
x = x.view(batch, -1)  # 刚创建的tensor肯定连续
```

✅ **需要保证共享内存**
```python
x = torch.randn(10, 20)
y = x.view(-1)  # 必须共享内存，否则报错提醒
```

✅ **性能敏感的代码**
```python
# 在循环中，view能保证不会意外复制
for data in dataloader:
    data = data.view(batch_size, -1)  # 快速且可预测
```

### 什么时候用 reshape

✅ **不确定tensor是否连续**
```python
def flexible_function(x):
    # x的来源不明确
    return x.reshape(target_shape)
```

✅ **快速原型开发**
```python
# 不想处理连续性问题，求方便
x = complicated_ops(x)
x = x.reshape(-1)
```

✅ **不关心是否共享内存**
```python
x = x.reshape(new_shape)  # 复制就复制，无所谓
```

### 推荐的最佳实践

```python
# ✅ 方式1: 根据场景选择
if x.is_contiguous():
    y = x.view(-1)      # 连续时用view
else:
    y = x.reshape(-1)   # 不连续时用reshape

# ✅ 方式2: 显式表达意图（推荐）
y = x.contiguous().view(-1)  # "我知道可能不连续，我处理了"

# ✅ 方式3: 简单粗暴
y = x.reshape(-1)  # 安全但可能隐藏性能问题

# ❌ 不推荐
y = x.view(-1)  # 不检查就用view，可能报错
```

---

## 4. 性能对比

```python
import torch
import time

x = torch.randn(1000, 1000)

# 连续tensor: view和reshape性能相同
start = time.time()
for _ in range(10000):
    _ = x.view(-1)
print(f"连续tensor - view: {time.time() - start:.4f}s")

start = time.time()
for _ in range(10000):
    _ = x.reshape(-1)
print(f"连续tensor - reshape: {time.time() - start:.4f}s")

# 不连续tensor: reshape需要复制，性能下降
y = x.t()  # 不连续

start = time.time()
for _ in range(10000):
    _ = y.reshape(-1)
print(f"不连续tensor - reshape: {time.time() - start:.4f}s")

start = time.time()
for _ in range(10000):
    _ = y.contiguous().view(-1)
print(f"不连续tensor - contiguous+view: {time.time() - start:.4f}s")
```

---

## 5. 总结

### 连续性
- **连续** = 内存存储顺序和逻辑顺序一致
- **不连续** = 通过stride实现的"视角变换"
- 转置、permute、某些切片会导致不连续

### view vs reshape
- **view**: 严格、快速、保证共享内存，但要求连续
- **reshape**: 灵活、安全、自动处理，但可能隐藏性能问题

### 记忆口诀
> **view是严格的老师，reshape是温柔的助手**
>
> - 用view时，代码会告诉你哪里有问题（不连续时报错）
> - 用reshape时，代码会帮你解决问题（自动复制）
>
> 两者都有价值，关键是在合适的场景用合适的工具！

### 快速决策树

```
需要重塑tensor形状？
├─ 明确知道tensor连续 → 用 view
├─ 不确定是否连续
│  ├─ 性能敏感 → 用 contiguous().view()
│  └─ 求方便 → 用 reshape
└─ 必须保证共享内存 → 用 view（不连续会报错提醒）
```
