---
title: PyTorch 自动微分机制
sidebar_label: 自动微分机制
date: 2025-01-13
last_update:
  date: 2025-01-13
---

# PyTorch 自动微分机制

## 什么是自动微分（Autograd）

自动微分是 PyTorch 的核心功能，它能够**自动计算梯度**，让我们无需手动推导和编写复杂的求导公式。这是实现神经网络反向传播的基础。

```python
import torch

# 创建需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3

# 自动计算梯度
z.backward()
print(x.grad)  # tensor([12.])
```

在这个例子中，PyTorch 自动计算了 dz/dx = 12，而我们不需要手动写出求导公式。

## 自动微分的实现原理

### 1. 核心机制：动态计算图

PyTorch 使用**动态计算图**记录所有操作，每个 Tensor 内部维护：

- `data`：实际的数据值
- `grad`：梯度值
- `grad_fn`：指向创建该 Tensor 的函数对象
- `requires_grad`：是否需要计算梯度

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3

# 计算图结构：
# x -> (pow) -> y -> (mul) -> z
#      ↑             ↑
#   grad_fn      grad_fn
```

### 2. 反向传播过程

每个操作都对应一个 Function 对象，包含 `forward()` 和 `backward()` 方法：

```python
# 简化的乘法反向传播实现
class MulBackward:
    def __init__(self, input1, input2):
        self.input1 = input1
        self.input2 = input2

    def backward(self, grad_output):
        # 链式法则：∂L/∂x = ∂L/∂z * ∂z/∂x
        grad_input1 = grad_output * self.input2
        grad_input2 = grad_output * self.input1
        return grad_input1, grad_input2
```

调用 `backward()` 时，PyTorch 按拓扑排序遍历计算图，从输出节点开始反向调用每个操作的梯度计算：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4
z = y * 3   # z = 12

z.backward()
# 反向传播计算：
# dz/dz = 1
# dy/dy = dz/dy = 1 * 3 = 3
# dx/dx = dy/dx = 3 * 2x = 3 * 4 = 12
```

### 3. 关键特性

**梯度累积**
```python
x = torch.tensor([1.0], requires_grad=True)

y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([2.])

y2 = x ** 3
y2.backward()  # 梯度会累积
print(x.grad)  # tensor([5.]) = 2 + 3

# 需要手动清零
x.grad.zero_()
```

**计算图释放**
```python
y = x ** 2
y.backward()  # 默认情况下，计算图会被释放

# 如果需要多次反向传播
y = x ** 2
y.backward(retain_graph=True)  # 保留计算图
```

## 动态图 vs 静态图

### 什么是动态图和静态图？

这是深度学习框架中两种不同的计算图构建方式：

- **静态图**（TensorFlow 1.x）：先定义后运行（Define-and-Run），先构建完整的计算图，再输入数据执行
- **动态图**（PyTorch）：边定义边运行（Define-by-Run），代码执行到哪里，图就构建到哪里

### 静态图示例（TensorFlow 1.x）

```python
import tensorflow as tf

# 1. 定义阶段：构建计算图
x = tf.placeholder(tf.float32)
y = x * 2

# 2. 运行阶段：在 Session 中执行
with tf.Session() as sess:
    result = sess.run(y, feed_dict={x: 3.0})
```

### 动态图示例（PyTorch）

```python
import torch

x = torch.tensor([3.0])
y = x * 2  # 这一行执行时，同时构建图和计算结果
```

### 对比差异

| 特性 | 静态图 | 动态图（PyTorch） |
|------|--------|---------|
| **构建时机** | 预先定义 | 运行时构建 |
| **灵活性** | 受限，需要特殊算子处理条件分支 | 高度灵活，可以用原生 Python 控制流 |
| **调试** | 困难，需要特殊工具 | 容易，可以用 Python 调试器 |
| **性能优化** | 更好，可以全局优化 | 较弱，但 PyTorch 2.0 引入了编译优化 |
| **学习曲线** | 陡峭 | 平缓 |

### 控制流的区别

**动态图（PyTorch）- 自然直观**
```python
def forward(x, n):
    result = x
    for i in range(n):  # n 可以是运行时变量
        if result.sum() > 0:
            result = result * 2
        else:
            result = result + 1
    return result
```

**静态图（TensorFlow 1.x）- 需要特殊算子**
```python
def forward(x, n):
    def body(i, result):
        result = tf.cond(
            tf.reduce_sum(result) > 0,
            lambda: result * 2,
            lambda: result + 1
        )
        return i + 1, result

    _, result = tf.while_loop(
        lambda i, _: i < n,
        body,
        [0, x]
    )
    return result
```

### 现代框架的融合

现在的框架都在结合两者的优点：

- **TensorFlow 2.x**：默认使用动态图（Eager Execution），可以用 `@tf.function` 装饰器转换为静态图优化
- **PyTorch 2.0**：引入 `torch.compile()`，将动态图编译优化为静态图

## 总结

- **自动微分**通过动态计算图和反向传播自动计算梯度，是 PyTorch 的核心机制
- **动态图**让 PyTorch 更灵活、易调试，适合研究和快速原型开发
- **静态图**性能更好，适合生产部署，现代框架通过编译技术融合两者优点
- PyTorch 的动态图优势：代码即模型，符合 Python 编程习惯，调试方便
