---
title: 梯度消失和爆炸问题
sidebar_label: 梯度消失和爆炸问题
date: 2025-01-17
last_update:
  date: 2025-02-03
---

import sigmoid_pic from '@site/static/img/sigmoid_pic.png'
import tanh_pic from '@site/static/img/tanh_pic.png'

# 梯度消失和爆炸问题

在深度神经网络中，梯度消失和梯度爆炸是两个常见的问题。这些问题会严重影响模型的训练效果。其中，tanh和sigmoid作为早期的激活函数，天然就存在梯度消失问题。从下图可以看到，当输入值x过大或过小时，这些激活函数的梯度会趋近于0，导致梯度消失。因此在使用这些激活函数时，需要特别注意以下几点：

1. 进行合适的数据预处理，将输入数据归一化到合适的范围
2. 使用批量归一化(Batch Normalization)等技术来缓解梯度问题
3. 考虑使用ReLU等更现代的激活函数作为替代方案

## Sigmoid函数

<img src={sigmoid_pic} width="50%" />

Sigmoid函数将输入映射到(0,1)区间，在历史上被广泛使用。但其存在以下问题：
- 容易发生梯度消失
- 输出不是零中心的
- 计算成本相对较高

### 解决办法

为了解决Sigmoid函数的这些问题，我们可以:

1. 使用ReLU等现代激活函数替代Sigmoid
2. 在必须使用Sigmoid的场景(如二分类输出层)，搭配以下技术:
   - 使用Batch Normalization
   - 合理初始化权重
   - 使用残差连接
   - 调整学习率
3. 在数据预处理阶段进行标准化，避免输入值过大或过小


## Tanh函数

<img src={tanh_pic} width="50%" />

Tanh函数将输入映射到(-1,1)区间，是Sigmoid的改进版本：
- 输出是零中心的，这有利于下一层的学习
- 但仍然存在梯度消失问题
- 计算开销相对较大

### 解决办法

为了解决Tanh函数的梯度消失问题，我们可以:

1. 使用ReLU等现代激活函数替代Tanh
2. 在必须使用Tanh的场景下，可以：
   - 应用Batch Normalization
   - 使用残差连接
   - 合理设置学习率
   - 谨慎初始化权重参数
3. 确保输入数据经过适当的预处理和归一化

## 梯度爆炸

### relu

ReLU激活函数虽然在正半轴上梯度恒为1，避免了梯度消失问题，但在深层神经网络中仍可能出现梯度爆炸。这主要是因为:

1. 权重初始化不当可能导致某些神经元的输出过大
2. 连续多层的正向传播会使信号被不断放大
3. 反向传播时梯度会呈指数级增长

为了解决这个问题，可以采取以下措施：

1. He初始化 - 根据输入维度合理初始化权重
2. 梯度裁剪 - 限制梯度的最大范围
3. 使用Batch Normalization - 在每一层标准化数据分布
4. 选择合适的学习率 - 避免参数更新步长过大
5. 使用残差连接 - 缓解深层网络的训练难度

```py
class DeepReLUNetwork(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1, 1) for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()
        self.gradients = []  # 存储中间梯度
        
        # 使用较大的权重初始化
        for layer in self.layers:
            # 错误的初始化 导致梯度爆炸问题
            nn.init.normal_(layer.weight, mean=1.5, std=0.1)
            nn.init.zeros_(layer.bias)

        # 正确的初始化
        # for layer in self.layers:
        #     nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        #     nn.init.constant_(layer.bias, 0.01)
        for i, layer in enumerate(model.layers):
            print(f"Layer {i+1} weight: {layer.weight.item():.4f}, bias: {layer.bias.item():.4f}")
        print("========")
    
    def forward(self, x):
        intermediates = []  # 存储中间结果
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)  # 保存每一层的输出
            x = self.relu(x)
        
        # 为每个中间结果注册hook
        for i, intermediate in enumerate(intermediates):
            intermediate.register_hook(lambda grad, idx=i: self.gradients.insert(0, grad.item()))  # 注意使用insert(0,)
        return x
```

## 代码

https://github.com/Hao-yiwen/deeplearning/blob/master/pytorch/week3/practise_9_relu_boom.ipynb
