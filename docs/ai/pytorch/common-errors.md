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

---

## 核心区别

### `w -= lr * w.grad`（✅ 正确）

* **in-place 操作**，不创建新张量
* `w` 还是原来的对象，`w.requires_grad` 依然是 `True`
* 只是数值改变了，下一轮仍会计算梯度

### `w = w - lr * w.grad`（❌ 错误）

* **创建新张量**，在 `no_grad()` 中计算
* 新张量的 `requires_grad` 是 `False`
* `w` 指向新张量后，下一轮不会再计算梯度

**更严重的陷阱：**

```python
# 如果 w 是 nn.Parameter
model.linear.weight = model.linear.weight - lr * model.linear.weight.grad
# ❌ 右边是普通 Tensor，不是 nn.Parameter
# ❌ 优化器和 model.parameters() 里找不到它了！
```

---

## 推荐写法

### ✅ 手写梯度下降

```python
for x, y in data:
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad  # 使用 in-place 操作
            p.grad.zero_()
```

### ✅ 使用优化器

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for x, y in data:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

---

## 常见报错

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation
```

**原因：** 没有用 `torch.no_grad()` 包裹 in-place 操作

```python
# ❌ 没有 no_grad()
w -= lr * w.grad  # 报错！
```

**解决：** 加上 `with torch.no_grad()`

---

## 一句话总结

* **`w -= lr * w.grad`**：原地改值，`w` 还是需要梯度的参数 ✅
* **`w = w - lr * w.grad`**：创建新张量，`requires_grad` 变 `False` ❌

---

## 相关文档

- [PyTorch 自动求导机制](./autograd-mechanism.md)
- [PyTorch 计算图详解](./computational_graph.md)
