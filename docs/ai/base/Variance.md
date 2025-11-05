# 方差 (Variance) 详解

## 什么是方差

方差是衡量数据**离散程度**的统计量，用来描述数据相对于平均值的波动大小。

- **方差大**：数据分散，波动大
- **方差小**：数据集中，波动小

**形象比喻：** 如果把数据看作是一群人站的位置，均值是他们的中心点，那么方差就是衡量这群人站得有多"散"。

## 数学定义

### 总体方差（σ²）
$$
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
$$

### 样本方差（s²）
$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

其中：
- **σ²** 或 **s²**：方差
- **μ** 或 **x̄**：均值（平均值）
- **N** 或 **n**：样本数量
- **xi**：第i个数据点

### 标准差（σ 或 s）
$$
\sigma = \sqrt{\sigma^2}
$$

**标准差是方差的平方根**，单位与原数据一致，更直观。

## 直观理解

假设有两组考试成绩：

```python
# 第一组：成绩比较集中
scores_1 = [85, 87, 86, 88, 84]  
# 均值: 86, 方差: 2, 标准差: 1.4

# 第二组：成绩很分散
scores_2 = [60, 95, 70, 100, 75]  
# 均值: 80, 方差: 256, 标准差: 16

# 第一组方差小 → 成绩稳定
# 第二组方差大 → 成绩波动大
```

## Python代码示例

### PyTorch计算方差

```python
import torch

data = torch.tensor([85., 87., 86., 88., 84.])

# 计算方差
var = torch.var(data)                    # 样本方差（默认 unbiased=True）
var_biased = torch.var(data, unbiased=False)  # 总体方差

# 计算标准差
std = torch.std(data)                    # 样本标准差
std_biased = torch.std(data, unbiased=False)  # 总体标准差

print(f"方差: {var:.2f}")
print(f"标准差: {std:.2f}")
```

## torch.normal - 正态分布采样

`torch.normal()` 是PyTorch中从**正态分布（高斯分布）**中采样的函数，在深度学习中应用非常广泛。

### 基本语法

```python
torch.normal(mean, std, size=None)
```

参数：
- **mean**：均值（μ）
- **std**：标准差（σ），注意是标准差不是方差！
- **size**：输出张量的形状

### 使用示例

#### 1. 基本用法

```python
import torch

# 从均值=0，标准差=1的正态分布中采样10个数
samples = torch.normal(mean=0.0, std=1.0, size=(10,))
print(samples)
# tensor([ 0.5410, -0.1734,  0.6699, -1.3270, ...])

# 验证
print(f"采样的均值: {samples.mean():.3f}")  # 接近0
print(f"采样的标准差: {samples.std():.3f}")  # 接近1
```

#### 2. 权重初始化（最常用）

```python
import torch.nn as nn

# 方式1: 直接使用torch.normal
weights = torch.normal(mean=0.0, std=0.01, size=(100, 50))

# 方式2: 使用nn.init（推荐）
linear = nn.Linear(100, 50)
nn.init.normal_(linear.weight, mean=0.0, std=0.01)

# 方式3: Xavier/He初始化（自动计算std）
nn.init.xavier_normal_(linear.weight)  # std = sqrt(2 / (fan_in + fan_out))
nn.init.kaiming_normal_(linear.weight)  # std = sqrt(2 / fan_in)
```

#### 3. 添加噪声

```python
# 给数据添加高斯噪声
clean_data = torch.randn(100, 10)
noise = torch.normal(mean=0.0, std=0.1, size=clean_data.shape)
noisy_data = clean_data + noise
```

#### 4. 不同形状的采样

```python
# 1D: 生成100个数
samples_1d = torch.normal(0, 1, size=(100,))

# 2D: 生成矩阵
samples_2d = torch.normal(0, 1, size=(10, 20))

# 3D: 生成三维张量
samples_3d = torch.normal(0, 1, size=(5, 10, 20))

# 4D: 用于卷积层（batch, channels, height, width）
samples_4d = torch.normal(0, 0.01, size=(32, 3, 224, 224))
```

#### 5. 每个元素不同的均值/标准差

```python
# 为每个位置指定不同的均值和标准差
mean_tensor = torch.tensor([0.0, 1.0, 2.0])
std_tensor = torch.tensor([0.1, 0.5, 1.0])

# 每个元素从不同的正态分布采样
samples = torch.normal(mean_tensor, std_tensor)
print(samples)
# tensor([0.05, 1.23, 3.14])  # 第1个从N(0, 0.1²)，第2个从N(1, 0.5²)，第3个从N(2, 1²)
```

## 在深度学习中的应用

### 1. 权重初始化

**为什么需要随机初始化？**
- 如果所有权重都是0，所有神经元会学到相同的特征（对称性问题）
- 合适的方差可以避免梯度消失/爆炸

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        
        # 使用正态分布初始化权重
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)
```

**常用初始化策略：**

| 方法 | 标准差公式 | 适用场景 |
|------|----------|---------|
| 简单正态分布 | std = 0.01 | 小网络、简单任务 |
| Xavier/Glorot | std = √(2/(fan_in + fan_out)) | Sigmoid、Tanh激活 |
| He/Kaiming | std = √(2/fan_in) | ReLU激活（常用） |

```python
# Xavier初始化
nn.init.xavier_normal_(layer.weight)

# He初始化（ReLU网络推荐）
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### 2. Dropout的替代 - 高斯噪声

```python
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            noise = torch.normal(0, self.std, size=x.shape, device=x.device)
            return x + noise
        return x

# 使用
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    GaussianNoise(std=0.1),  # 添加噪声
    nn.Linear(50, 10)
)
```

### 3. 数据增强

```python
def add_gaussian_noise(images, std=0.1):
    """给图像添加高斯噪声"""
    noise = torch.normal(0, std, size=images.shape)
    return images + noise

# 使用
noisy_images = add_gaussian_noise(clean_images, std=0.05)
```

### 5. 批归一化（Batch Normalization）

Batch Normalization的核心思想就是将每层的输入标准化到均值为0、方差为1：

```python
# 简化的BN实现
def batch_norm_simple(x, eps=1e-5):
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm
```

## 方差 vs 标准差

| 特性 | 方差 (σ²) | 标准差 (σ) |
|------|----------|----------|
| 单位 | 原始单位的平方 | 与原始数据相同 |
| 数值 | 通常较大 | 相对较小 |
| 直观性 | 较差 | 更直观 |
| 计算 | 基础统计量 | 方差的平方根 |
| PyTorch | `torch.var()` | `torch.std()` |

**实践建议：**
- 理论推导常用方差（数学上更方便）
- 实际使用常用标准差（更直观）
- **torch.normal()使用的是标准差，不是方差！**

## 常见问题

### Q1: torch.normal 和 torch.randn 的区别？

```python
# torch.randn: 固定从N(0,1)采样
x = torch.randn(10, 20)  # 均值0，标准差1

# torch.normal: 可以指定任意均值和标准差
y = torch.normal(mean=5.0, std=2.0, size=(10, 20))  # 均值5，标准差2
```

### Q2: 为什么要使用正态分布初始化？

1. **中心极限定理**：多个随机变量的和趋向正态分布
2. **数学性质好**：可微、对称、易于分析
3. **经验有效**：大量实验证明效果好
4. **避免对称性**：打破神经元之间的对称性

### Q3: 方差太大或太小会怎样？

```python
# 方差太小 → 权重接近0 → 信息传递弱
weights_small = torch.normal(0, 0.001, size=(100, 100))

# 方差太大 → 梯度爆炸
weights_large = torch.normal(0, 10, size=(100, 100))

# 合适的方差（He初始化）
std = np.sqrt(2.0 / 100)  # fan_in = 100
weights_good = torch.normal(0, std, size=(100, 100))
```

## 实战示例：完整的模型初始化

```python
import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        # 使用He初始化（适用于ReLU）
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.size(1)
                # 解释：Why 2/fan_in?
                # 因为ReLU激活函数只让一半输出有用，为了让每层输出的方差保持不变，
                # He初始化采用 Var(w) = 2 / fan_in，通过std=√(2/fan_in)保证前向和反向传播的信号不过强/弱，避免梯度消失/爆炸
                std = np.sqrt(2.0 / fan_in)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                
                # 偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型
model = MyNeuralNetwork(784, 256, 10)

# 验证初始化
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name}:")
        print(f"  均值: {param.mean().item():.6f}")
        print(f"  标准差: {param.std().item():.6f}")
```

## 总结

**方差的核心概念：**
- 方差衡量数据的离散程度
- 标准差 = √方差，更直观
- 正态分布由均值和方差完全确定

**torch.normal的要点：**
- 用于从正态分布采样
- 参数是**标准差**（不是方差！）
- 在权重初始化、添加噪声等场景广泛使用
- 合适的方差对模型训练至关重要

**实践建议：**
- ReLU网络使用He初始化：`nn.init.kaiming_normal_`
- Sigmoid/Tanh使用Xavier初始化：`nn.init.xavier_normal_`
- 自定义时，标准差通常在0.01-0.1之间
- 记住：torch.normal使用的是std（标准差），不是var（方差）