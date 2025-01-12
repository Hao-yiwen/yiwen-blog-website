# 计算图

计算图(Computational Graph)是一个有向无环图(DAG)，用于表示计算过程中各个操作和数据之间的依赖关系。在深度学习中，它特别重要，因为：

1. 基本结构

- 节点：表示操作(operations)或变量(variables)
- 边：表示数据流动的方向
- 每个节点保存：
    - 前向计算的结果
    - 反向传播需要的中间值
    - 对输入的偏导数计算方法

2. 举个简单例子：

```py
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y    # 乘法操作
w = z + x    # 加法操作
w.backward() # 反向传播
print(x.grad)
```

```
x --→ (*) --→ z --→ (+) --→ w
      ↑            ↑
y ----┘      x ----┘
```

## 梯度为什么要清零

1. 梯度累加
```py
# 不清零的情况
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y * 3   

# 第一次反向传播
z.backward()  
print(x.grad)  # tensor([6.])

# 第二次反向传播
z.backward()
print(x.grad)  # tensor([12.]) # 6 + 6，梯度累加了！
```
2. 为什么梯度会累加
- PyTorch中梯度是累积的，不会自动清零
- 每次backward()调用都会将新计算的梯度加到已有的梯度上
- 这个设计是为了支持累积多个batch的梯度