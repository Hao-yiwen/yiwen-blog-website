---
title: PyTorch CPU-GPU 异步执行机制
sidebar_label: CPU-GPU 异步执行
date: 2025-11-09
last_update:
  date: 2025-11-09
---

# PyTorch CPU-GPU 异步执行机制

## 核心概念

PyTorch 的 GPU 操作是**异步**的：
- **CPU**：发送指令（"指挥官"）
- **GPU**：执行计算（"执行者"）
- **关键**：CPU 发完指令立即返回，不等待 GPU 完成

### 类比

```
同步模式（慢）：点一道菜 → 等厨师做完 → 点下一道 → 等待...
异步模式（快）：一次性点完所有菜 → 离开，厨师慢慢做
```

## CPU 和 GPU 分别做什么？

```python
import torch
import torch.nn as nn

model = MyModel().cuda()
data = torch.randn(32, 10).cuda()

# CPU 的工作：发送指令
for epoch in range(100):              # CPU 控制循环
    for batch in dataloader:          # CPU 控制循环
        optimizer.zero_grad()
        output = model(data)          # CPU 发送指令，立即返回
        loss = criterion(output)      # CPU 发送指令，立即返回
        loss.backward()               # CPU 发送指令，立即返回
        optimizer.step()              # CPU 发送指令，立即返回
        # CPU 继续下一个循环，GPU 在后台干活

# GPU 的工作：执行实际计算
# GPU 收到指令后，依次执行：forward → loss → backward → step
```

**时间线**：
```
CPU:  [发指令] [发指令] [发指令] → 立即继续循环
        ↓         ↓         ↓
GPU:         [forward] [loss]   [backward]  ← 后台执行
```

## 触发同步的操作

### ✅ 不触发同步（快）

```python
# 纯 GPU 操作，CPU 只发指令
output = model(data)              # 异步
loss = criterion(output)          # 异步
loss.backward()                   # 异步
optimizer.step()                  # 异步
```

### ❌ 触发同步（慢）

```python
# 需要 CPU 读取 GPU 的值，必须等待
scalar = loss.item()              # 同步！CPU 等待 GPU
print(loss)                       # 同步！内部调用 .item()
if loss < 0.1:                    # 同步！需要读取值
    break

cpu_tensor = gpu_tensor.cpu()     # 同步！GPU → CPU 传输
numpy_array = gpu_tensor.numpy()  # 同步！转换为 NumPy
torch.cuda.synchronize()          # 强制同步
```

## 为什么训练循环不慢？

### ✅ 训练循环（异步，快）

```python
for epoch in range(100):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)        # 异步
        loss = criterion(output)    # 异步
        loss.backward()             # 异步
        optimizer.step()            # 异步
```

**原因**：
- CPU 的 for 循环只是发送指令，不等待结果
- GPU 接收指令后持续工作
- CPU 和 GPU 并行运行，不互相等待

### ❌ 模型内控制流（同步，极慢）

```python
class SlowModel(nn.Module):
    def forward(self, x):
        x = self.fc1(x)
        # ❌ 每次循环都要读取 GPU 的值
        while x.sum().item() > 1:   # 极慢！
            x = x / 2
        return self.fc2(x)

# 时间线：
# CPU: [发指令] [等GPU] [读值] [发指令] [等GPU] [读值] ...
# GPU:         [计算]   → 返回  [计算]   → 返回 ...
# CPU 和 GPU 互相等待，利用率极低！
```

**原因**：
- 每次循环都调用 `.item()`，触发同步
- 同步次数 = batch_size × 循环次数
- 32 个样本，每个循环 10 次 = 320 次同步！

## 代码对比

### ✅ 每 N 个 batch 打印（可接受）

```python
for i, (data, target) in enumerate(dataloader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output)
    loss.backward()
    optimizer.step()

    # 每 100 个 batch 才同步一次
    if i % 100 == 0:
        print(f"Loss: {loss.item()}")  # 可接受
```

### ❌ 每个 batch 都打印（慢）

```python
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output)

    print(f"Loss: {loss.item()}")  # ❌ 每次都同步，慢！

    loss.backward()
    optimizer.step()
```

### ✅ 正确的模型写法

```python
class FastModel(nn.Module):
    def forward(self, x):
        x = self.fc1(x)
        # ✅ 固定次数的循环，不依赖 GPU 的值
        for _ in range(10):
            x = x / 2
        return self.fc2(x)
```

## 性能对比

| 场景 | 同步频率 | 性能影响 |
|------|---------|---------|
| 训练循环控制流 | 低（epoch级） | ✅ 可忽略 |
| 每 100 batch 打印 | 很低 | ✅ 可接受 |
| 每个 batch 打印 | 中等 | ⚠️ 略慢（2-4倍） |
| 模型内数据依赖控制流 | 极高 | ❌ 极慢（10-30倍） |

## 性能测试

```python
import torch
import time

model = MyModel().cuda()
data = torch.randn(128, 1000).cuda()

# 测试 1：异步执行（快）
start = time.time()
for _ in range(1000):
    output = model(data)
    loss = criterion(output)
    loss.backward()
torch.cuda.synchronize()
print(f"异步: {time.time() - start:.2f}s")  # ~0.8s

# 测试 2：每次同步（慢）
start = time.time()
for _ in range(1000):
    output = model(data)
    loss = criterion(output)
    _ = loss.item()  # ❌ 触发同步
    loss.backward()
torch.cuda.synchronize()
print(f"同步: {time.time() - start:.2f}s")  # ~3.4s（慢 4 倍）
```

## 最佳实践

### ✅ 推荐

1. **训练循环**：放心使用 for/while/if
   ```python
   for epoch in range(100):        # CPU 控制，不慢
       for data in loader:          # CPU 控制，不慢
           # 训练代码
   ```

2. **模型 forward**：避免数据依赖的控制流
   ```python
   # ✅ 固定次数
   for _ in range(10):
       x = layer(x)

   # ❌ 依赖 GPU 值
   while x.sum().item() > 1:
       x = layer(x)
   ```

3. **调试打印**：每 N 个 batch 一次
   ```python
   if batch_idx % 100 == 0:
       print(f"Loss: {loss.item()}")
   ```

### ❌ 避免

- 训练循环内频繁打印 `loss.item()`
- 模型内的数据依赖控制流（`while x.mean().item() > 0.5`）
- 不必要的 CPU-GPU 传输（`.cpu()`, `.numpy()`）

## 关键结论

### 为什么训练循环不慢？
> **训练循环虽然在 CPU 运行，但只是发送指令，不等待 GPU 完成，所以不慢。**

```
CPU 循环：for i in range(1000)  ← 很快，只发指令
  ↓
GPU 队列：[指令1] [指令2] ... [指令1000]  ← GPU 慢慢执行
```

### 为什么模型内控制流很慢？
> **模型内部的数据依赖控制流需要频繁读取 GPU 结果，导致大量同步等待。**

```
CPU: while x.sum().item() > 1  ← 每次都要读 GPU 的值
  ↓
GPU: 算完 sum → 返回 → 等待 CPU → 算除法 → 返回 → 等待...

CPU 和 GPU 互相等待，都闲着！
```

## 总结

| 概念 | 说明 |
|------|------|
| **异步执行** | CPU 发送指令后立即返回，不等待 GPU |
| **同步操作** | CPU 必须等待 GPU 返回结果（`.item()`, `.cpu()`） |
| **训练循环** | CPU 控制，但只发指令，不慢 |
| **模型控制流** | 依赖 GPU 数据会频繁同步，极慢 |

**记住**：CPU 发指令很快，等待 GPU 返回结果很慢！
