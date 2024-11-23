# 反向传播算法

2024年诺贝尔物理学奖杯幸顿教授获得，获得的原因是反向传播算法促成了多层次神经网络的搭建成为可能。所以什么是反向传播算法，他的意义到底是什么。

## 说明

个人理解：反向传播是通过从输出层到输入层求的输出相对于各个层次参数的链式求导过程。

gpt理解: 反向传播是计算损失函数关于网络参数的梯度的算法。通过链式法则，反向传播从输出层开始，逐层向前计算梯度，以更新网络的权重和偏置，最小化损失函数。

## 过程

```py
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name="layer1"),
        Dense(1, activation='sigmoid', name="layer2")
    ]
)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              )

model.fit(Xt, yt, epochs=10, verbose=1)
```

在多层次的模型训练过程中，需要不断降低损失函数，也就是说需要不断朝着梯度降低的过程进行。

而进行中的非常重要一环就是损失函数对于各个层次的参数的梯度计算，反向传播算法使得神经网络学习中损失函数和各层次的参数进行绑定，使得结果忘最优化过程进行，从而使得各个隐藏层最终都能获取到非常有用的特征。
