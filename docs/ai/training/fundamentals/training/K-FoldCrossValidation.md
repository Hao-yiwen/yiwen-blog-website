---
title: "K折交叉验证 (K-Fold Cross Validation)"
sidebar_label: "K折交叉验证 (K-Fold Cross Validation)"
date: 2025-11-06
last_update:
  date: 2025-11-05
---

# K折交叉验证 (K-Fold Cross Validation)

## 什么是K折验证

K折交叉验证是机器学习中一种常用的模型评估方法。它将数据集分成K个大小相等的子集（折），然后进行K次训练和验证，每次使用不同的折作为验证集，其余作为训练集。最后将K次结果取平均，得到模型的综合性能评估。

## 为什么需要K折验证

**传统的单次划分问题：**
- 验证集的选择可能存在偶然性
- 某些特殊样本可能正好都在训练集或验证集中
- 小数据集上单次划分结果不稳定

**K折验证的优势：**
- 每个样本都会被用作验证集一次，更全面
- 充分利用有限的数据
- 评估结果更可靠、更稳定

## 工作原理

假设有100个样本，使用5折交叉验证：

```
原始数据: [样本1, 样本2, 样本3, ..., 样本100]

第1折: 训练[21-100] → 验证[1-20]   → 准确率 85%
第2折: 训练[1-20, 41-100] → 验证[21-40] → 准确率 87%
第3折: 训练[1-40, 61-100] → 验证[41-60] → 准确率 86%
第4折: 训练[1-60, 81-100] → 验证[61-80] → 准确率 88%
第5折: 训练[1-80] → 验证[81-100] → 准确率 84%

最终评估: (85% + 87% + 86% + 88% + 84%) / 5 = 86%
```

## 步骤详解

1. **划分数据**：将数据集随机打乱后平均分成K份
2. **循环训练**：进行K轮训练
   - 第i轮：用第i折作为验证集，其余K-1折作为训练集
   - 训练模型并在验证集上评估
3. **汇总结果**：计算K次评估指标的平均值和标准差

## 代码示例

### PyTorch + K折验证

```python
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# 假设已有 dataset 和 model
dataset = ...  # 你的数据集
model_class = ...  # 你的模型类

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"\n训练第 {fold + 1} 折...")
    
    # 创建数据加载器
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    
    # 重新初始化模型
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = nn.CrossEntropyLoss()(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    fold_results.append(accuracy)
    print(f"第 {fold + 1} 折准确率: {accuracy:.3f}")

print(f"\n平均准确率: {np.mean(fold_results):.3f} (+/- {np.std(fold_results):.3f})")
```

## K值的选择

| K值 | 训练集大小 | 计算成本 | 适用场景 |
|-----|----------|---------|---------|
| K=5 | 80% | 中等 | 最常用，平衡了偏差和方差 |
| K=10 | 90% | 较高 | 数据量中等时的标准选择 |
| K=N (LOOCV) | 99.9% | 很高 | 数据量很小时（&lt;100样本） |
| K=3 | 66% | 较低 | 快速实验或大数据集 |

**一般建议：**
- 数据量小（&lt;1000）：K=10
- 数据量中等（1000-10000）：K=5
- 数据量大（&gt;10000）：K=3 或直接用单次划分

## 变体

### 1. 分层K折（Stratified K-Fold）
保持每折中各类别的比例与原数据集一致，适合**类别不平衡**的数据：

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)
```

### 2. 留一法（Leave-One-Out, LOO）
K = 样本数，每次只用一个样本作为验证集：

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

### 3. 时间序列交叉验证
保持时间顺序，避免用未来数据预测过去：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
```

## 优缺点

### 优点
✓ 充分利用数据，每个样本都参与训练和验证  
✓ 评估结果更稳定可靠  
✓ 减少因数据划分带来的偶然性  
✓ 适合小数据集  

### 缺点
✗ 计算成本是单次划分的K倍  
✗ 大数据集上可能不实用  
✗ 每折模型独立训练，无法获得最终用于预测的模型  

## 注意事项

1. **数据泄露**：划分必须在任何预处理（如标准化）之前进行，或在每折内部独立处理

```python
# ❌ 错误：在划分前标准化
scaler.fit(X)  # 用了全部数据！
X_scaled = scaler.transform(X)
kfold.split(X_scaled)

# ✅ 正确：每折内部独立标准化
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    scaler.fit(X_train)  # 只用训练集
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
```

2. **随机性**：设置 `shuffle=True` 和 `random_state` 以保证可重复性

3. **不用于最终训练**：K折验证是为了评估模型性能，最终部署时应该用全部数据训练

4. **时间序列数据**：不要随机划分，使用 `TimeSeriesSplit`

## 实际应用场景

- **模型选择**：比较不同算法的性能
- **超参数调优**：与网格搜索结合使用
- **特征工程评估**：评估新特征是否有效
- **小数据集评估**：数据少时更可靠的评估方法

## 总结

K折交叉验证是机器学习中评估模型性能的金标准方法。它通过多次训练和验证，给出更全面、更可靠的性能估计。虽然计算成本较高，但在数据有限或需要精确评估时非常值得使用。

**记住：**
- K=5 或 K=10 是最常用的选择
- 类别不平衡时使用分层K折
- 时间序列数据要保持时间顺序
- 避免数据泄露