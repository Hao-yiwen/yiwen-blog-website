---
title: PyTorch CPU-GPU 异步执行与同步
sidebar_label: CPU-GPU 异步执行
date: 2025-11-09
last_update:
  date: 2025-11-09
---

# PyTorch CPU-GPU 异步执行与同步

## 核心概念

PyTorch 的 GPU 操作是**异步**的：
- **CPU**：发送指令（"指挥官"）
- **GPU**：执行计算（"执行者"）
- CPU 发完指令立即返回，不等待 GPU 完成

### 类比：餐厅点餐

```
同步模式（慢）：
顾客（CPU）→ 下单 → 等待厨师做完 → 拿到菜 → 下第二单 → 等待...

异步模式（快）：
顾客（CPU）→ 快速下完所有单 → 离开
厨师（GPU）→ 按顺序依次做菜
```

## 异步执行机制

```python
import torch
import torch.nn as nn

model = MyModel().cuda()
data = torch.randn(32, 3, 224, 224).cuda()

# CPU 发送指令，不等待 GPU
output = model(data)      # CPU: "GPU，计算 forward"，立即返回
loss = criterion(output)  # CPU: "GPU，计算 loss"，立即返回
loss.backward()           # CPU: "GPU，反向传播"，立即返回
optimizer.step()          # CPU: "GPU，更新参数"，立即返回

# GPU 在后台并行执行所有操作
```

### 时间线图解

```
CPU:  [发指令] [发指令] [发指令] [发指令] → 立即进入下一个循环
        ↓         ↓         ↓         ↓
GPU:           [forward] [loss]   [backward] [step]  ← 后台执行
```

**关键点**：CPU 不等待 GPU 完成就继续执行下一行代码！

## CPU 和 GPU 分别什么时候运行？

### CPU 的工作

```python
# 这些都在 CPU 上运行
for epoch in range(100):              # CPU 控制循环
    for data, target in dataloader:   # CPU 控制循环
        # CPU 只是"发送指令"给 GPU
        optimizer.zero_grad()
        output = model(data)          # CPU 发送指令后立即返回
        loss = criterion(output)      # CPU 发送指令后立即返回
        loss.backward()               # CPU 发送指令后立即返回
        optimizer.step()              # CPU 发送指令后立即返回

        # CPU 继续下一个循环，GPU 在后台干活
```

### GPU 的工作

```python
# 这些在 GPU 上实际执行计算
model = model.cuda()                  # 模型参数在 GPU 上
data = data.cuda()                    # 数据在 GPU 上

output = model(data)                  # GPU 执行矩阵运算
loss = criterion(output, target)      # GPU 计算损失
loss.backward()                       # GPU 计算梯度
optimizer.step()                      # GPU 更新参数
```

### 完整流程示例

```python
import torch
import torch.nn as nn

# 1. CPU 准备模型和数据
model = nn.Linear(10, 5).cuda()       # CPU: 创建模型，发送到 GPU
data = torch.randn(32, 10).cuda()     # CPU: 创建数据，发送到 GPU

# 2. CPU 发送计算指令
output = model(data)                  # CPU: 发送指令，立即返回
                                      # GPU: 开始计算（CPU 已经不等了）

# 3. CPU 继续发送下一个指令
loss = output.sum()                   # CPU: 再发一个指令，立即返回
                                      # GPU: 等 forward 完再计算 sum

# 4. CPU 继续发送反向传播指令
loss.backward()                       # CPU: 又发一个指令，立即返回
                                      # GPU: 等 sum 完再反向传播

# 此时：CPU 已经执行完所有代码，GPU 可能还在计算最后的 backward
```

## 触发同步的操作

有些操作会**强制 CPU 等待 GPU 完成**，这叫做**同步**。

### ✅ 不触发同步（快）

```python
# 纯 GPU 操作，CPU 只发指令
x = x.cuda()                          # CPU 发送传输指令
y = model(x)                          # CPU 发送计算指令
loss = criterion(y, target)           # CPU 发送计算指令
loss.backward()                       # CPU 发送计算指令
optimizer.step()                      # CPU 发送计算指令

# CPU 一气呵成发完指令，GPU 慢慢算
```

### ❌ 触发同步（慢）

```python
# 1. 读取 GPU 上的标量值
scalar = loss.item()                  # ❌ CPU 必须等 GPU 返回值
print(loss)                           # ❌ 内部调用 .item()
if loss < 0.1:                        # ❌ 需要读取 loss 的值
    break

# 2. GPU → CPU 数据传输
cpu_tensor = gpu_tensor.cpu()         # ❌ 必须等 GPU 计算完
numpy_array = gpu_tensor.numpy()      # ❌ 需要先传回 CPU

# 3. 显式同步
torch.cuda.synchronize()              # ❌ 强制等待所有 GPU 操作完成
```

### 为什么这些操作触发同步？

```python
loss = criterion(output, target)      # CPU 发送指令，loss 只是"承诺"
print(loss.item())                    # CPU 需要实际的值，必须等 GPU 算完

# 时间线：
# CPU: [发指令] [print] ← 被迫等待
#                ↓
# GPU:         [计算loss] → 返回值 ← CPU 在这里等待
```

## 代码对比：快 vs 慢

### ✅ 训练循环（异步，快）

```python
import torch
import torch.nn as nn
import time

model = MyModel().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

start = time.time()
for epoch in range(100):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)        # 异步
        loss = criterion(output)    # 异步
        loss.backward()             # 异步
        optimizer.step()            # 异步
        # GPU 持续工作，CPU 不等待

# 测量时需要同步一次
torch.cuda.synchronize()
print(f"训练时间: {time.time() - start:.2f}s")
```

**为什么不慢？**
- CPU 的 for 循环只是发送指令
- GPU 接收到指令后持续工作
- CPU 不等待 GPU 返回结果

### ❌ 每次打印 loss（同步，慢）

```python
for epoch in range(100):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output)

        # ❌ 每个 batch 都触发同步
        print(f"Loss: {loss.item()}")  # 慢！

        loss.backward()
        optimizer.step()
```

**为什么慢？**
```
Batch 1:
CPU: [发指令] [等待GPU] [打印] [发指令] [等待GPU] ...
GPU:         [计算loss] → 返回

Batch 2:
CPU: [发指令] [等待GPU] [打印] [发指令] [等待GPU] ...
GPU:         [计算loss] → 返回

每个 batch 都有一次等待，累积起来很慢！
```

### ✅ 每 N 个 batch 打印一次（可接受）

```python
for epoch in range(100):
    for i, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output)
        loss.backward()
        optimizer.step()

        # ✅ 每 100 个 batch 才同步一次
        if i % 100 == 0:
            print(f"Loss: {loss.item()}")  # 可接受
```

### ❌ 模型内控制流（同步，极慢）

```python
class SlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)

        # ❌ 每个样本、每次循环都触发同步
        while x.sum().item() > 1:   # 极慢！
            x = x / 2

        return self.fc2(x)

# 时间线：
# CPU: [发fc1指令] [等GPU] [读sum] [发除法指令] [等GPU] [读sum] ...
# GPU:           [算fc1] → [等CPU] [算除法] → [等CPU] ...
# CPU 和 GPU 互相等待，利用率极低！
```

**为什么极慢？**
- 每次循环都要读取 `x.sum().item()`
- CPU-GPU 同步次数 = 样本数 × 循环次数
- 32 个样本，每个循环 10 次 = 320 次同步！

### ✅ 正确做法：避免数据依赖的控制流

```python
class FastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc1(x)

        # ✅ 固定次数的循环，GPU 可以批量执行
        for _ in range(10):  # 不依赖 GPU 的值
            x = x / 2

        return self.fc2(x)

# 时间线：
# CPU: [发fc1] [发10次除法] [发fc2] → 立即返回
# GPU:       [算fc1] [算10次除法] [算fc2]  ← 一气呵成
```

## 性能影响对比

| 场景 | 同步频率 | 性能影响 | 原因 |
|------|---------|---------|------|
| 训练循环控制流 | 低（epoch/batch级） | ✅ 可忽略 | CPU 只发指令，不等待 |
| 每 100 个 batch 打印 | 很低 | ✅ 可接受 | 同步次数少 |
| 每个 batch 打印 loss | 中等 | ⚠️ 略慢 | 同步次数 = batch 数 |
| 模型内数据依赖控制流 | 极高 | ❌ 极慢（10-30倍） | 同步次数 = 样本数 × 循环次数 |

## 性能测试示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().cuda()
data = torch.randn(128, 1000).cuda()
target = torch.randn(128, 1000).cuda()

# 测试 1：正常训练（异步）
torch.cuda.synchronize()  # 确保之前的操作完成
start = time.time()
for _ in range(1000):
    output = model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
torch.cuda.synchronize()  # 等待所有操作完成再计时
print(f"异步执行: {time.time() - start:.2f}s")  # ~0.8s

# 测试 2：每次读取 loss（同步）
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    output = model(data)
    loss = F.mse_loss(output, target)
    _ = loss.item()  # ❌ 每次都触发同步
    loss.backward()
torch.cuda.synchronize()
print(f"每次同步: {time.time() - start:.2f}s")  # ~3.4s（慢 4 倍）

# 测试 3：模型内数据依赖控制流（极慢）
class SlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 1000)

    def forward(self, x):
        for _ in range(10):
            x = self.fc(x)
            if x.sum().item() > 1000:  # ❌ 数据依赖
                x = x / 2
        return x

slow_model = SlowModel().cuda()
torch.cuda.synchronize()
start = time.time()
for _ in range(100):  # 只跑 100 次，因为太慢了
    output = slow_model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
torch.cuda.synchronize()
print(f"模型内同步: {time.time() - start:.2f}s")  # ~15s（慢 20 倍！）
```

## 如何检查是否触发同步？

### 方法 1：使用 `torch.cuda.Event` 测量

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(data)
# 如果这里有同步操作，会看到明显的时间差
end.record()

torch.cuda.synchronize()
print(f"GPU 时间: {start.elapsed_time(end):.2f}ms")
```

### 方法 2：使用 profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA
    ],
    with_stack=True
) as prof:
    for _ in range(10):
        output = model(data)
        loss = criterion(output)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 最佳实践

### ✅ 推荐做法

1. **训练循环**：放心使用 for/while/if，不影响性能
   ```python
   for epoch in range(100):        # CPU 控制，不慢
       for data, target in loader:  # CPU 控制，不慢
           # 训练代码
   ```

2. **模型 forward**：避免数据依赖的控制流
   ```python
   # ✅ 好：固定次数
   for _ in range(10):
       x = layer(x)

   # ❌ 坏：依赖 GPU 值
   while x.sum().item() > 1:
       x = layer(x)
   ```

3. **调试打印**：每 N 个 batch 打印一次
   ```python
   if batch_idx % 100 == 0:
       print(f"Loss: {loss.item()}")
   ```

4. **验证指标**：在验证阶段读取值
   ```python
   model.eval()
   with torch.no_grad():
       for data, target in val_loader:
           output = model(data)
           loss = criterion(output)
           total_loss += loss.item()  # 验证阶段可以接受
   ```

### ❌ 避免的做法

1. **训练循环内频繁打印**
   ```python
   for data, target in loader:
       loss = criterion(output)
       print(loss.item())  # ❌ 每个 batch 都同步
   ```

2. **模型内的数据依赖控制流**
   ```python
   def forward(self, x):
       while x.mean().item() > 0.5:  # ❌ 极慢
           x = self.layer(x)
   ```

3. **不必要的 CPU-GPU 传输**
   ```python
   for data, target in loader:
       data_cpu = data.cpu()  # ❌ 不必要的同步
       data_gpu = data_cpu.cuda()
   ```

## 调试技巧

### 检测同步点

```python
import torch.cuda.profiler as profiler
import torch.autograd.profiler as autograd_profiler

# 启用 profiler
with autograd_profiler.profile(use_cuda=True) as prof:
    output = model(data)
    loss = criterion(output)
    loss.backward()

# 查看 CPU-GPU 同步点
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 强制同步来测量真实时间

```python
import time

# ❌ 错误的测量方式（测量的是发送指令的时间）
start = time.time()
output = model(data)
print(f"Time: {time.time() - start}")  # 几乎是 0

# ✅ 正确的测量方式
start = time.time()
output = model(data)
torch.cuda.synchronize()  # 等待 GPU 完成
print(f"Time: {time.time() - start}")  # 真实的计算时间
```

## 关键结论

### CPU 做什么？
- 控制训练循环（for/while/if）
- 发送 GPU 计算指令
- 管理数据加载
- **不等待 GPU 完成**（异步）

### GPU 做什么？
- 执行矩阵运算
- 计算损失和梯度
- 更新参数
- 按照 CPU 发送的指令顺序执行

### 为什么训练循环不慢？
> **训练循环虽然在 CPU 运行，但只是发送指令，不等待 GPU 完成，所以不慢。**

```
CPU 循环：for i in range(1000)  ← 很快，只是发指令
  ↓
GPU 队列：[指令1] [指令2] [指令3] ... [指令1000]  ← GPU 慢慢执行
```

### 为什么模型内控制流很慢？
> **模型内部的数据依赖控制流需要频繁读取 GPU 结果，导致大量同步等待。**

```
CPU: while x.sum().item() > 1  ← 每次都要读 GPU 的值
  ↓
GPU: 算完 sum → 返回给 CPU → 等待 CPU 发下一个指令 → ...

CPU 和 GPU 互相等待，都闲着！
```

## 总结

| 概念 | 说明 |
|------|------|
| **异步执行** | CPU 发送指令后立即返回，不等待 GPU |
| **同步操作** | CPU 必须等待 GPU 返回结果 |
| **训练循环** | CPU 控制，但只发指令，不慢 |
| **模型控制流** | 如果依赖 GPU 数据，会频繁同步，极慢 |
| **最佳实践** | 避免模型内的数据依赖控制流 |

记住：**CPU 发指令很快，等待 GPU 返回结果很慢！**
