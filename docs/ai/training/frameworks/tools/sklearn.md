---
title: "Scikit-learn 实用工具指南"
sidebar_label: "Scikit-learn 实用工具指南"
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# Scikit-learn 实用工具指南

## 介绍

✓ **数据预处理**：标准化、编码、缺失值处理  
✓ **数据划分**：train_test_split、K折交叉验证  
✓ **评估指标**：准确率、混淆矩阵、分类报告  
✓ **实用工具**：降维、特征选择等  

**核心理念：** sklearn负责数据处理和评估，PyTorch负责模型训练。

## 安装

```bash
pip install scikit-learn
```


## 最常用的工具（按使用频率）

## 1. 数据划分

### train_test_split - 最常用

```python
from sklearn.model_selection import train_test_split
import torch
import numpy as np

# 假设你有数据
X = np.random.randn(1000, 10)  # 1000个样本，10个特征
y = np.random.randint(0, 2, 1000)  # 二分类标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20%作为测试集
    random_state=42,    # 固定随机种子
    stratify=y          # 保持类别比例（分类任务推荐）
)

# 转换为PyTorch张量
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

print(f"训练集: {X_train.shape}")  # (800, 10)
print(f"测试集: {X_test.shape}")   # (200, 10)
```

### 三次划分：训练集、验证集、测试集

```python
# 方法1: 两次调用train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
# 结果：训练集70%，验证集15%，测试集15%

# 方法2: 手动计算比例
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)
# 结果：训练集60%，验证集20%，测试集20%
```

### 与PyTorch DataLoader配合

```python
from torch.utils.data import TensorDataset, DataLoader

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建Dataset
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

## 2. 数据预处理

### StandardScaler - 标准化（最常用）

将数据转换为均值0、标准差1的分布：

```python
from sklearn.preprocessing import StandardScaler
import torch

# 创建标准化器
scaler = StandardScaler()

# 在训练集上拟合
scaler.fit(X_train)

# 转换训练集和测试集
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚠️ 重要：不要对测试集调用fit！
# ❌ 错误：scaler.fit(X_test)
# ✅ 正确：只用transform

# 转换为PyTorch张量
X_train_t = torch.FloatTensor(X_train_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
```

**为什么需要标准化？**
- 不同特征的量纲可能差异很大
- 加速梯度下降收敛
- 某些层（如BatchNorm）需要稳定的输入分布

**何时使用：**
- ✓ 线性层、全连接网络
- ✓ 特征量纲差异大
- ✗ CNN处理图像（图像通常已归一化到[0,1]）
- ✗ 已经做了BatchNorm的网络（可选）

### MinMaxScaler - 归一化到[0,1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 自定义范围
scaler = MinMaxScaler(feature_range=(-1, 1))
```

### RobustScaler - 对异常值鲁棒

```python
from sklearn.preprocessing import RobustScaler

# 使用中位数和四分位数，对异常值不敏感
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 保存和加载Scaler

```python
import joblib

# 保存scaler
joblib.dump(scaler, 'scaler.pkl')

# 加载scaler
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

---

## 3. 类别编码

### LabelEncoder - 标签编码

将类别标签转换为整数：

```python
from sklearn.preprocessing import LabelEncoder

# 原始标签（字符串）
labels = ['cat', 'dog', 'cat', 'bird', 'dog']

# 编码
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
print(y_encoded)  # [1 2 1 0 2]

# 查看类别映射
print(le.classes_)  # ['bird' 'cat' 'dog']

# 反向解码
y_decoded = le.inverse_transform(y_encoded)
print(y_decoded)  # ['cat' 'dog' 'cat' 'bird' 'dog']

# 转为PyTorch张量
y_tensor = torch.LongTensor(y_encoded)
```

### OneHotEncoder - 独热编码

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 类别特征
categories = np.array(['red', 'blue', 'green', 'red']).reshape(-1, 1)

# 独热编码
encoder = OneHotEncoder(sparse=False)
one_hot = encoder.fit_transform(categories)
print(one_hot)
# [[0. 0. 1.]   # red
#  [1. 0. 0.]   # blue
#  [0. 1. 0.]   # green
#  [0. 0. 1.]]  # red

# 转为PyTorch张量
one_hot_t = torch.FloatTensor(one_hot)
```

### 实际场景：混合特征编码

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 假设有混合类型的数据
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 80000],
    'city': ['NYC', 'LA', 'NYC', 'SF'],
    'label': ['A', 'B', 'A', 'B']
})

# 1. 编码类别特征
le_city = LabelEncoder()
df['city_encoded'] = le_city.fit_transform(df['city'])

le_label = LabelEncoder()
y = le_label.fit_transform(df['label'])

# 2. 提取数值特征
X_numeric = df[['age', 'income', 'city_encoded']].values

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 4. 转为PyTorch
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.LongTensor(y)
```

---

## 4. 评估指标（必备）

### 分类指标

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import numpy as np

# PyTorch模型预测
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    y_pred_t = outputs.argmax(dim=1)

# 转为numpy数组
y_true = y_test  # 或 y_test_t.numpy()
y_pred = y_pred_t.cpu().numpy()

# 1. 准确率
acc = accuracy_score(y_true, y_pred)
print(f"准确率: {acc:.3f}")

# 2. 精确率、召回率、F1
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"精确率: {precision:.3f}")
print(f"召回率: {recall:.3f}")
print(f"F1分数: {f1:.3f}")

# 3. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵:")
print(cm)

# 4. 完整分类报告
report = classification_report(y_true, y_pred)
print(report)
```

### 可视化混淆矩阵

```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.savefig('confusion_matrix.png')
plt.show()
```

### 回归指标

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np

# 获取预测值
model.eval()
with torch.no_grad():
    y_pred_t = model(X_test_t)

y_true = y_test
y_pred = y_pred_t.cpu().numpy()

# MSE 和 RMSE
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R² 分数
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
```

---

## 5. 交叉验证（配合PyTorch）

### K折交叉验证

```python
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np

# 准备数据
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# 创建K折
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n训练第 {fold + 1} 折...")
    
    # 划分数据
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 标准化（每折独立）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 转为张量
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.LongTensor(y_val)
    
    # 重新初始化模型
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = val_outputs.argmax(dim=1)
        accuracy = (val_preds == y_val_t).float().mean().item()
    
    fold_results.append(accuracy)
    print(f"第 {fold + 1} 折准确率: {accuracy:.3f}")

print(f"\n平均准确率: {np.mean(fold_results):.3f} (+/- {np.std(fold_results):.3f})")
```

### 分层K折（类别不平衡时）

```python
from sklearn.model_selection import StratifiedKFold

# 保持每折中各类别的比例
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    # 训练代码...
    pass
```

### 与DataLoader配合

```python
from torch.utils.data import TensorDataset, DataLoader, Subset

# 创建完整数据集
dataset = TensorDataset(
    torch.FloatTensor(X),
    torch.LongTensor(y)
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    # 创建子集
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # 创建DataLoader
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # 训练...
    for epoch in range(10):
        for batch_X, batch_y in train_loader:
            # 训练代码
            pass
```

## 快速参考

### 数据划分
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### 标准化
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # 训练集
X_test_scaled = scaler.transform(X_test)  # 测试集（只transform）
```

### 评估
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
```

### K折验证
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

---

## 常见问题

### Q1: 为什么测试集不能fit？

```python
# ❌ 错误 - 会导致数据泄露
scaler.fit(X_test)

# ✅ 正确 - 只在训练集上fit
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

**原因：** fit会计算统计量（均值、标准差），如果在测试集上fit，相当于"偷看"了测试数据的信息。

### Q2: 每折交叉验证需要重新fit scaler吗？

**需要！** 每折的训练集都不同，应该独立fit：

```python
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    
    # 每折独立创建和fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
```

### Q3: 图像数据需要StandardScaler吗？

通常不需要。图像一般直接除以255归一化到[0,1]：

```python
# 图像数据
X_images = X_images / 255.0

# 或者标准化到[-1, 1]
X_images = (X_images / 255.0 - 0.5) * 2
```

### Q4: 如何在新数据上使用保存的scaler？

```python
# 训练时保存
joblib.dump(scaler, 'scaler.pkl')

# 推理时加载
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

---

## 总结

**sklearn在PyTorch项目中的角色：**

| 阶段 | sklearn工具 | PyTorch负责 |
|------|------------|------------|
| 数据准备 | train_test_split, StandardScaler | Dataset, DataLoader |
| 训练 | KFold（交叉验证） | 模型定义、训练循环 |
| 评估 | accuracy_score, confusion_matrix | 模型推理 |

**记住这些要点：**
1. ✓ 只在训练集上fit，测试集只transform
2. ✓ K折验证时每折独立fit scaler
3. ✓ 保存scaler以便推理时使用
4. ✓ 类别不平衡时使用stratify参数
5. ✓ 使用sklearn做评估，简单又准确