---
title: 前向传播与反向传播算法详解
sidebar_label: 前向与反向传播
date: 2024-11-23
last_update:
  date: 2024-12-19
---

# 前向传播与反向传播算法详解

2024 年诺贝尔物理学奖授予了杰弗里·辛顿（Geoffrey Hinton）教授等人，反向传播（Backpropagation）算法正是让深层神经网络训练成为现实的核心基石。理解神经网络训练，首先要厘清**前向传播**与**反向传播**这两个相辅相成的阶段。

## 一、前向传播（Forward Propagation）

前向传播是指输入数据从输入层依次流经各隐藏层、最终到达输出层生成预测结果的过程：

1. **线性变换**：每一层接收上一层的输出，乘以权重矩阵 $W$ 并加上偏置向量 $b$（即 $z = Wx + b$）。
2. **非线性激活**：通过激活函数（如 ReLU、GELU、Sigmoid）引入非线性表达能力（即 $a = \sigma(z)$）。
3. **计算损失**：在输出层得到预测值 $\hat{y}$ 后，通过损失函数 $L(\hat{y}, y)$ 度量预测值与真实标签之间的误差。

## 二、反向传播（Backpropagation）

反向传播是**利用微积分链式法则（Chain Rule），从输出层向输入层逐层反向计算损失函数对各层参数梯度**的高效算法：

- **直观理解**：根据最终预测误差，沿着网络反向追责，算出每一层的权重 $W$ 和偏置 $b$ 对总误差各自“贡献”了多少斜率（偏导数 $\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$）。
- **参数更新**：优化器（如 SGD、AdamW）拿到梯度后，沿梯度反方向更新各层参数（$W \leftarrow W - \eta \frac{\partial L}{\partial W}$），使损失函数逐步逼近极小值。

## 三、代码示例与训练闭环

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation="sigmoid", name="layer1"),
    Dense(1, activation="sigmoid", name="layer2"),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

# 内部自动循环执行：前向传播算 Loss -> 反向传播求梯度 -> Adam 更新权重
model.fit(Xt, yt, epochs=10, verbose=1)
```

正是通过“前向计算预测误差 + 反向链式传播梯度 + 优化器迭代更新”的闭环，深层神经网络中的各个隐藏层才能自动学习并提取出有效的层级特征表示。
