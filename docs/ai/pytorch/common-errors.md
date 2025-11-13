# PyTorch 常见错误与陷阱

## 梯度更新：in-place 操作 vs 重新赋值

### 问题

**为什么在 `torch.no_grad()` 中，`w -= lr * w.grad` 和 `w = w - lr * w.grad` 的行为完全不同？**

```python
# 方式1：原地操作
with torch.no_grad():
    w -= lr * w.grad  # ✅ 推荐

# 方式2：重新赋值
with torch.no_grad():
    w = w - lr * w.grad  # ❌ 错误！w.requires_grad 会变成 False
```

### 核心区别

**这两种写法一个是原地改值，一个是重新创建了一个张量**，在 `torch.no_grad()` 里行为完全不同。

---

## 1. `w -= lr * w.grad` 发生了什么？（✅ 正确方式）

```python
with torch.no_grad():
    w -= lr * w.grad
```

**行为分析：**

* 这是 **in-place（原地）操作**，不会创建新的张量
* 变量 `w` 还是原来的那个 `w`，所以：
  * `w.requires_grad` 依然是 `True`
  * 只是把数据从 `w.data` 改成了更新后的数值
* 因为包在 `torch.no_grad()` 里，这个更新不会被 autograd 记录到计算图里，但 **参数本身还是"可求梯度"的参数**
* 下一轮前向 + `backward()` 仍然会给它算梯度

**这是手写梯度下降时推荐的写法**（或者用官方的 `optimizer.step()`）。

---

## 2. `w = w - lr * w.grad` 发生了什么？（❌ 错误方式）

```python
with torch.no_grad():
    w = w - lr * w.grad
```

**行为分析：**

* 这里是 **新建一个张量**：
  1. 先在 `no_grad` 环境中算出 `tmp = w - lr * w.grad`
  2. 再让 Python 变量名 `w` 指向这个新的 `tmp`
* 由于是在 `torch.no_grad()` 里算的，**这个新张量的 `requires_grad` 会是 `False`**
* 导致的问题：
  * `w.requires_grad` 变成了 `False`
  * 下一次就不会再给它算梯度了
  * **不是梯度消失，而是梯度参数变为 False！**

**更严重的陷阱：**

如果 `w` 原本是 `nn.Module` 里的 `nn.Parameter`，你用这种方式重新赋值：

```python
model.linear.weight = model.linear.weight - lr * model.linear.weight.grad
```

那右边得到的是个普通 `Tensor`，不是 `nn.Parameter`，**优化器和 `model.parameters()` 里都找不到它了**，训练直接"废掉"。

---

## 3. 为什么书上说不会梯度消失，还会报错？

### 典型报错

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

**这个报错只会在「没用 `torch.no_grad()` 的 in-place」更新里出现**：

```python
# ❌ 没有 no_grad()
w -= lr * w.grad   # ← 这里就会报上面那个错
```

因为 autograd 不允许你对"叶子张量（参数）"做会被记录到计算图里的原地修改。

### 正确理解

**只要你真的写在 `with torch.no_grad():` 里面，`w -= lr * w.grad` 是不会报这个错的。**

如果你遇到这个报错，要么是：

* `with torch.no_grad()` 缩进没对上，其实没包住
* 要么某一步用了 `w.data` 等其他方式，搞乱了 autograd 的状态

---

## 4. 实际写法建议

### ✅ 推荐写法：手写梯度下降

```python
for x, y in data:
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad  # ← 使用 in-place 操作
            p.grad.zero_()
```

### ✅ 推荐写法：使用优化器

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for x, y in data:
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()  # ← 内部使用 in-place 操作
```

### ❌ 错误写法

```python
# 错误1：重新赋值
with torch.no_grad():
    for p in model.parameters():
        p = p - lr * p.grad  # ❌ 创建新张量，p.requires_grad 变 False

# 错误2：没有 no_grad 包裹
for p in model.parameters():
    p -= lr * p.grad  # ❌ 报错：leaf Variable in-place operation
```

---

## 一句话总结

* **`w -= lr * w.grad`**：原地改值，`w` 还是那个需要梯度的参数，只是数值变了，这是我们想要的 ✅
* **`w = w - lr * w.grad`**：新建了一个不需要梯度的张量，再把 `w` 指向它，导致后面 `w` 就不再参与梯度计算 ❌

---

## 相关概念

### In-place 操作的特点

```python
# In-place 操作（修改自身）
x += 1
x.add_(1)
x.relu_()
x.fill_(0)

# 非 In-place 操作（返回新张量）
y = x + 1
y = x.add(1)
y = x.relu()
y = torch.zeros_like(x)
```

**规律**：
* 带下划线后缀的方法（如 `add_`, `relu_`）通常是 in-place
* 复合赋值运算符（`+=`, `-=`, `*=`, `/=`）是 in-place
* 普通运算符（`+`, `-`, `*`, `/`）返回新张量

### 为什么需要 `torch.no_grad()`？

```python
# 没有 no_grad - 会记录计算图
loss.backward()
with torch.no_grad():
    w -= lr * w.grad  # ← 不记录这步到计算图

# 如果不用 no_grad
loss.backward()
w -= lr * w.grad  # ← autograd 会报错！
```

在参数更新时：
* 我们不需要对"参数更新"这个操作求梯度
* 用 `no_grad()` 可以节省内存和计算
* 避免 autograd 对 leaf variable 的 in-place 操作报错

---

## 相关文档

- [PyTorch 自动求导机制](./autograd-mechanism.md)
- [PyTorch 计算图详解](./computational_graph.md)
