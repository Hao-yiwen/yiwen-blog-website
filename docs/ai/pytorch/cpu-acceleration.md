---
title: PyTorch CPU 加速效果详解
sidebar_label: CPU 加速效果
date: 2025-11-09
last_update:
  date: 2025-11-09
---

# PyTorch CPU 加速效果详解

## 核心结论

是的！PyTorch 在 CPU 上也有明显加速效果。

相比纯 Python 代码，PyTorch (即使在 CPU 上) 也能有**几十到上百倍**的加速。

---

## 实际测试对比

```python
import torch
import time
import numpy as np

n = 10000

# 1. 纯 Python (最慢)
def pure_python_matmul(a, b):
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

# 2. NumPy (快)
a_np = np.random.randn(n, n)
b_np = np.random.randn(n, n)

start = time.time()
c_np = np.dot(a_np, b_np)
print(f"NumPy (CPU): {time.time() - start:.4f}秒")

# 3. PyTorch CPU (差不多快)
a_torch = torch.randn(n, n)
b_torch = torch.randn(n, n)

start = time.time()
c_torch = torch.mm(a_torch, b_torch)
print(f"PyTorch (CPU): {time.time() - start:.4f}秒")

# 4. PyTorch GPU (超快！)
if torch.cuda.is_available():
    a_gpu = a_torch.cuda()
    b_gpu = b_torch.cuda()

    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    print(f"PyTorch (GPU): {time.time() - start:.4f}秒")
```

### 典型结果 (10000×10000 矩阵乘法)

```
纯 Python:      ~30分钟 (太慢了，一般不会真跑完)
NumPy (CPU):    ~2秒
PyTorch (CPU):  ~2秒
PyTorch (GPU):  ~0.05秒
```

---

## 为什么 PyTorch CPU 也快？

### 1. 底层用 C++ 实现

```python
# 看起来是 Python，实际调用 C++ 代码
result = torch.mm(a, b)  # 底层是高度优化的 C++
```

### 2. 使用优化的数学库

- **Intel MKL** (Math Kernel Library)
- **OpenBLAS**
- 这些库经过几十年优化，用了 SIMD 指令等

### 3. 向量化操作

```python
# 纯 Python: 逐元素循环 (慢)
for i in range(len(a)):
    c[i] = a[i] + b[i]

# PyTorch: 向量化 (快)
c = a + b  # 一次性处理整个数组
```

### 4. 多线程并行

PyTorch 会自动使用多个 CPU 核心

---

## PyTorch CPU vs NumPy

### CPU 性能对比

**性能差不多**，因为：
- 都用类似的底层库 (BLAS, LAPACK)
- 都是向量化操作
- 都支持多线程

### ⚠️ 关键区别：GPU 支持

**NumPy 的限制：**
- ❌ **只支持 CPU**，完全不支持 GPU 加速
- ❌ 即使有 GPU，NumPy 也无法使用
- ❌ 大规模矩阵运算受限于 CPU 性能

**PyTorch 的优势：**
- ✅ 同时支持 CPU 和 GPU
- ✅ 代码几乎不用改，只需 `.cuda()` 或 `.to(device)`
- ✅ GPU 可以带来 50-100 倍的加速

```python
import numpy as np
import torch

# NumPy: 只能在 CPU 上运行
a_np = np.random.randn(10000, 10000)
b_np = np.random.randn(10000, 10000)
c_np = np.dot(a_np, b_np)  # ❌ 无法使用 GPU，约 2 秒

# PyTorch: 可以选择 CPU 或 GPU
a_torch = torch.randn(10000, 10000)
b_torch = torch.randn(10000, 10000)

# CPU 模式（和 NumPy 差不多）
c_cpu = torch.mm(a_torch, b_torch)  # 约 2 秒

# GPU 模式（快得多！）
if torch.cuda.is_available():
    a_gpu = a_torch.cuda()
    b_gpu = b_torch.cuda()
    c_gpu = torch.mm(a_gpu, b_gpu)  # ✅ 约 0.05 秒，快 40 倍！
```

### 📌 选择建议

| 场景 | 推荐工具 | 原因 |
|------|---------|------|
| 只做数值计算，数据量小 | NumPy | 更轻量，生态成熟 |
| 需要训练神经网络 | PyTorch | 自动微分，灵活 |
| **需要 GPU 加速** | **PyTorch** | **NumPy 不支持 GPU** |
| 大规模矩阵运算 | PyTorch (GPU) | 性能远超 NumPy |
| 科学计算（SciPy 生态） | NumPy | 工具链完善 |

### 💡 从 NumPy 迁移到 PyTorch

如果你的 NumPy 代码需要 GPU 加速，迁移很简单：

```python
# NumPy 代码
import numpy as np
a = np.random.randn(1000, 1000)
b = np.random.randn(1000, 1000)
c = np.dot(a, b)
result = np.sum(c)

# 等价的 PyTorch 代码（CPU）
import torch
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)
c = torch.mm(a, b)
result = torch.sum(c)

# 一行代码切换到 GPU
a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()
c = torch.mm(a, b)  # 自动在 GPU 上运行
result = torch.sum(c)
```

**关键点：**
- PyTorch 的 API 和 NumPy 非常相似（`np.dot` → `torch.mm`）
- NumPy 不支持 GPU，如果需要 GPU 加速必须用 PyTorch 或其他框架
- PyTorch 在 CPU 上性能和 NumPy 相当，但在 GPU 上快几十倍

---

## 总结

| | 纯 Python | NumPy/PyTorch (CPU) | PyTorch (GPU) |
|---|---|---|---|
| 矩阵乘法 | 100倍时间 | 1倍时间 (基准) | **50-100倍更快** |
| 实现 | Python 循环 | C++/Fortran | CUDA |

**结论**: PyTorch 在 CPU 上也比纯 Python 快得多，但 GPU 才是真正的"核武器" 🚀

---

## 什么时候用 CPU，什么时候用 GPU？

### ✅ 适合用 CPU 的场景

1. **数据量小**
   ```python
   # 小模型 + 小数据，GPU 不划算
   x = torch.randn(32, 10)  # 32 样本，10 特征
   model = nn.Linear(10, 1)  # 只有 11 个参数
   ```

2. **调试代码**
   ```python
   # 开发阶段用 CPU 更方便
   model = MyModel()  # 不用 .cuda()
   output = model(data)
   ```

3. **不支持 GPU 的操作**
   ```python
   # 某些操作只能在 CPU 上运行
   cpu_data = gpu_data.cpu()
   numpy_array = cpu_data.numpy()
   ```

### ✅ 适合用 GPU 的场景

1. **大模型训练**
   ```python
   model = TransformerModel(layers=12, hidden=768)  # 百万参数
   data = torch.randn(128, 512, 768)  # 大 batch size
   ```

2. **批量推理**
   ```python
   # 一次处理大量数据
   images = torch.randn(1000, 3, 224, 224).cuda()
   predictions = model(images)
   ```

3. **矩阵密集运算**
   ```python
   # 卷积、全连接等密集运算
   conv = nn.Conv2d(64, 128, 3).cuda()
   x = torch.randn(32, 64, 224, 224).cuda()
   output = conv(x)  # GPU 快 50-100 倍
   ```

---

## 最佳实践

### 1. 开发时用 CPU，训练时用 GPU

```python
# 使用 device 参数，方便切换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = data.to(device)
```

### 2. 小心 CPU-GPU 传输开销

```python
# ❌ 频繁传输，很慢
for data in dataloader:
    data = data.cuda()  # 每次都传输，慢
    output = model(data)

# ✅ 使用 pin_memory 加速
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)
```

### 3. 混合使用

```python
# 复杂控制流在 CPU，密集计算在 GPU
if some_condition:  # CPU 判断
    x = x.cuda()
    x = heavy_computation(x)  # GPU 计算
    x = x.cpu()  # 传回 CPU
```

---

## 参考资料

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Intel MKL for PyTorch](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-math-kernel-library.html)
