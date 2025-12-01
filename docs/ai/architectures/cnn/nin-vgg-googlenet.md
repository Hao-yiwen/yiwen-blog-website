---
title: CNN经典架构：NiN、VGG、GoogleNet
sidebar_label: NiN、VGG、GoogleNet
date: 2025-01-12
last_update:
  date: 2025-01-12
---

# CNN经典架构：NiN、VGG、GoogleNet

## 发展时间线

```
AlexNet (2012) → NiN (2013) → VGG/GoogleNet (2014) → ResNet (2015)
```

这三个经典CNN架构都是在AlexNet之后、ResNet之前的重要里程碑，它们各自引入了影响深远的创新思想。

---

## 1. NiN (Network in Network, 2013)

### 核心思想
**用 1×1 卷积增强局部感受野的非线性表达能力**

### 创新点

#### 1×1 卷积层（重要创新！）

```python
# 传统卷积后直接激活
conv = nn.Conv2d(64, 128, kernel_size=3)
out = F.relu(conv(x))

# NiN：卷积后加 1×1 卷积（mlpconv层）
conv = nn.Conv2d(64, 128, kernel_size=3)
nin1 = nn.Conv2d(128, 128, kernel_size=1)  # 1×1 卷积
nin2 = nn.Conv2d(128, 128, kernel_size=1)  # 再一个 1×1
out = F.relu(nin2(F.relu(nin1(F.relu(conv(x))))))
```

**1×1 卷积的作用**：
- 跨通道信息融合（在同一空间位置，混合不同通道特征）
- 增加非线性（多加几个ReLU）
- 降维/升维（改变通道数）

#### 全局平均池化（Global Average Pooling, GAP）

```python
# 传统做法：展平 + 全连接
x = x.view(x.size(0), -1)  # 展平
x = nn.Linear(7*7*512, 4096)(x)  # 巨大的全连接层
x = nn.Linear(4096, 1000)(x)

# NiN 做法：全局平均池化
x = F.adaptive_avg_pool2d(x, (1, 1))  # 每个通道平均成一个值
x = x.view(x.size(0), -1)              # 展平，无需全连接
```

**GAP 的优势**：
- **大幅减少参数**：AlexNet 最后全连接层占总参数90%+
- **防止过拟合**：没有那么多参数需要拟合
- **保持空间信息**：每个通道对应一个类别的"置信度图"

### 网络结构示意

```
输入图像 (224×224×3)
   ↓
[Conv 11×11 → 1×1 Conv → 1×1 Conv] + ReLU + MaxPool
   ↓
[Conv 5×5 → 1×1 Conv → 1×1 Conv] + ReLU + MaxPool
   ↓
[Conv 3×3 → 1×1 Conv → 1×1 Conv] + ReLU + MaxPool
   ↓
[Conv 3×3 → 1×1 Conv → 1×1 Conv] + ReLU
   ↓
Global Average Pooling (1×1×1000)
   ↓
Softmax
```

### 影响
- **开创了 1×1 卷积的先河**，后续网络（如GoogleNet、ResNet）大量使用
- **GAP 成为标配**，几乎所有现代网络都用

---

## 2. VGG (2014)

### 核心思想
**更深的网络 + 简单重复的小卷积核**

### 创新点

#### 全部使用 3×3 小卷积

```python
# 为什么用 3×3 而不是 7×7？

# 两个 3×3 卷积的感受野 = 一个 5×5
# 三个 3×3 卷积的感受野 = 一个 7×7

# 但参数量：
# 一个 7×7：C × C × 7 × 7 = 49C²
# 三个 3×3：3 × (C × C × 3 × 3) = 27C²

# 结论：3×3 更少参数，更多非线性（3个ReLU vs 1个）
```

#### 深度优先策略
VGG-16/VGG-19：通过堆叠简单模块达到很深的深度

### 网络结构（VGG-16）

```python
VGG16 = [
    # Block 1: 64 通道
    Conv3×3(64) → Conv3×3(64) → MaxPool,

    # Block 2: 128 通道
    Conv3×3(128) → Conv3×3(128) → MaxPool,

    # Block 3: 256 通道
    Conv3×3(256) → Conv3×3(256) → Conv3×3(256) → MaxPool,

    # Block 4: 512 通道
    Conv3×3(512) → Conv3×3(512) → Conv3×3(512) → MaxPool,

    # Block 5: 512 通道
    Conv3×3(512) → Conv3×3(512) → Conv3×3(512) → MaxPool,

    # 全连接层
    FC(4096) → FC(4096) → FC(1000)
]
```

**特点**：
- **规律性强**：全是 3×3 卷积 + 2×2 池化
- **通道数翻倍**：64 → 128 → 256 → 512
- **容易理解和实现**

### 代码示例

```python
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        # 每个block：若干3×3卷积 + MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # ... 更多 block
```

### 优缺点

**优点**：
- ✅ 结构简单，容易理解和实现
- ✅ 证明了"更深的网络 = 更好的性能"
- ✅ 迁移学习效果好（特征通用性强）

**缺点**：
- ❌ 参数量巨大（VGG-16：138M 参数，主要在全连接层）
- ❌ 计算量大，训练慢
- ❌ 内存占用高

---

## 3. GoogleNet / Inception v1 (2014)

### 核心思想
**多尺度特征融合 + 降低计算量**

### 创新点：Inception 模块

```
                    输入
                     |
        +------------+------------+------------+
        |            |            |            |
     1×1 Conv    1×1 Conv    1×1 Conv    3×3 MaxPool
        |            |            |            |
        |         3×3 Conv    5×5 Conv     1×1 Conv
        |            |            |            |
        +------------+------------+------------+
                     |
              Filter Concatenation
```

**设计思路**：
1. **多尺度并行**：同时使用 1×1、3×3、5×5 卷积，捕获不同尺度特征
2. **1×1 降维**：在 3×3 和 5×5 前先用 1×1 卷积降维（减少计算量）
3. **拼接融合**：将所有分支输出在通道维度拼接

### Inception 模块代码

```python
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__()
        # 分支1：1×1 卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 分支2：1×1 降维 → 3×3 卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 分支3：1×1 降维 → 5×5 卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 分支4：3×3 MaxPool → 1×1 卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 通道维度拼接
```

### 1×1 降维的作用（重要！）

```python
# 不降维：直接 5×5 卷积
# 输入：256 通道，输出：256 通道
conv = nn.Conv2d(256, 256, kernel_size=5, padding=2)
# 参数量：256 × 256 × 5 × 5 = 1,638,400

# 降维：1×1 降到 64 → 5×5 卷积到 256
conv1 = nn.Conv2d(256, 64, kernel_size=1)    # 256×64×1×1 = 16,384
conv2 = nn.Conv2d(64, 256, kernel_size=5, padding=2)  # 64×256×5×5 = 409,600
# 总参数：16,384 + 409,600 = 425,984

# 节省：(1,638,400 - 425,984) / 1,638,400 = 74% ！
```

### GoogleNet 整体结构

```
输入 (224×224×3)
   ↓
Conv + MaxPool                    # 初始卷积
   ↓
Conv + MaxPool                    # 降维
   ↓
Inception(3a) → Inception(3b) → MaxPool
   ↓
Inception(4a) → Inception(4b) → Inception(4c) → Inception(4d) → Inception(4e) → MaxPool
   ↓
Inception(5a) → Inception(5b)
   ↓
Global Average Pooling            # 借鉴 NiN
   ↓
FC(1000) → Softmax
```

### 辅助分类器（训练技巧）

GoogleNet 中间层还添加了两个辅助分类器：

```python
# 在中间层（如 Inception 4a 输出）添加
AuxClassifier = nn.Sequential(
    nn.AvgPool2d(5, 3),
    nn.Conv2d(512, 128, 1),
    nn.Flatten(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(1024, 1000)
)

# 总损失 = 主分类器损失 + 0.3 × 辅助分类器损失
```

**作用**：
- 缓解梯度消失（给中间层额外的梯度信号）
- 正则化效果
- **注意**：推理时不使用，只在训练时用

### 优缺点

**优点**：
- ✅ 参数少（7M vs VGG的138M）
- ✅ 计算效率高（1×1 降维）
- ✅ 多尺度特征融合
- ✅ 性能好（2014 ILSVRC 冠军）

**缺点**：
- ❌ 结构复杂，调参困难
- ❌ 实现比 VGG 复杂得多

---

## 三者对比总结

| 特性 | NiN (2013) | VGG (2014) | GoogleNet (2014) |
|------|-----------|-----------|-----------------|
| **核心思想** | 1×1卷积 + GAP | 深度 + 小卷积 | 多尺度 + 降维 |
| **深度** | 较浅 (7-8层) | 很深 (16-19层) | 很深 (22层) |
| **参数量** | 小 | **巨大** (138M) | **很小** (7M) |
| **卷积核** | 多样 | **全 3×3** | 多尺度并行 |
| **主要创新** | 1×1卷积, GAP | 证明深度价值 | Inception模块 |
| **计算效率** | 高 | 低 | **高** |
| **实现难度** | 中 | **简单** | 复杂 |
| **影响力** | 开创性技术 | 结构范式 | 效率典范 |

## 历史意义

### 1. NiN
- 发明了 1×1 卷积（后续网络必备）
- 提出 GAP 替代全连接层

### 2. VGG
- 证明"更深 = 更好"
- 简单重复结构成为经典范式
- 预训练模型广泛用于迁移学习

### 3. GoogleNet
- 平衡性能和效率的典范
- Inception 模块启发后续设计（如 Inception v2/v3/v4）
- 证明精心设计可以大幅减少参数

## 对后续的影响

- **ResNet (2015)**：借鉴 VGG 的深度思想 + 残差连接突破更深网络
- **MobileNet/ShuffleNet**：借鉴 1×1 卷积和降维思想，追求极致效率
- **EfficientNet**：综合深度、宽度、分辨率的平衡设计

这三个网络奠定了现代 CNN 的基础架构思想！
