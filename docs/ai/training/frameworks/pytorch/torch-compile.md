---
title: PyTorch 2.0 torch.compile
sidebar_label: torch.compile
date: 2025-01-13
last_update:
  date: 2025-01-13
---

# PyTorch 2.0 torch.compile

## 什么是 torch.compile？

`torch.compile` 是 PyTorch 2.0 的核心特性，它能够**在保持动态图编程体验的同时，获得静态图的性能优势**。

```python
import torch
import torch.nn as nn

# 普通的 PyTorch 模型（动态图）
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)

# 一行代码启用编译优化
compiled_model = torch.compile(model)

# 使用方式完全相同
x = torch.randn(32, 100)
output = compiled_model(x)  # 自动获得性能提升！
```

## 工作原理

`torch.compile` 的核心思想是**将动态图转换为优化的静态图执行**：

**1. 第一次运行时捕获操作序列**

```python
@torch.compile
def forward(x):
    x = x * 2      # 记录这个操作
    x = x + 1      # 记录这个操作
    return x.relu()  # 记录这个操作

# 第一次执行时，PyTorch 会：
# 1. 正常执行代码
# 2. 同时记录所有操作
# 3. 构建并优化静态图
```

**2. 后续运行直接执行优化后的静态图**

```python
# 之后的调用直接运行编译好的代码
# 不再需要 Python 解释器参与
result = forward(x)  # 快！
```

## 性能提升来源

### 1. 算子融合（Operator Fusion）

```python
# 原始代码：三个独立操作
x = x * 2
x = x + 1
x = x.relu()

# 编译后：融合成一个操作
x = fused_mul_add_relu(x, 2, 1)  # 一次 GPU kernel 调用

# 优势：
# - 减少内存读写（3 次 -> 1 次）
# - 减少 kernel 启动开销
# - 提高 GPU 利用率
```

### 2. 消除 Python 开销

```python
# 动态图：每次循环都有 Python 开销
for i in range(1000):
    x = x + 1  # 1000 次 Python 函数调用

# 静态图：编译成底层循环
# 直接执行，没有 Python 参与
```

### 3. 内存优化

```python
# 编译器可以分析整个计算流程
# 智能复用中间变量的内存
a = x * 2      # 分配 buffer_1
b = a + 1      # 分配 buffer_2
c = b.relu()   # 复用 buffer_1（a 不再使用）
```

## 性能对比

```python
import time
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = SimpleModel().cuda()
compiled_model = torch.compile(model)

x = torch.randn(128, 1024).cuda()

# 动态图性能
start = time.time()
for _ in range(100):
    y = model(x)
print(f"动态图: {time.time() - start:.3f}s")

# 静态图性能（第一次会有编译开销）
start = time.time()
for _ in range(100):
    y = compiled_model(x)
print(f"静态图: {time.time() - start:.3f}s")

# 典型结果：静态图可以快 1.5-2 倍！
```

## 支持动态控制流

```python
@torch.compile
def dynamic_forward(x, threshold):
    if x.sum() > threshold:  # 原生 Python 控制流
        return x * 2
    else:
        return x * 3

# 第一次运行针对具体的分支编译
# 如果条件改变，会重新编译
```

## 使用建议

**适合的场景**：
- 模型结构固定，需要多次推理
- 生产环境的模型部署
- 性能敏感的训练任务

**不适合的场景**：
- 只运行一两次（编译开销大于收益）
- 模型结构频繁变化
- 快速实验和调试阶段

## 总结

`torch.compile` 让你可以：
- ✅ 用动态图的方式写代码（灵活、易调试）
- ✅ 获得静态图的执行性能（快！）
- ✅ 一行代码就能启用优化

这就是 PyTorch 2.0 的核心优势——**两全其美**！
