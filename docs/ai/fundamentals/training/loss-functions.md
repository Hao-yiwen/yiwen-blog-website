---
title: PyTorch Loss 函数详解
sidebar_position: 20
tags: [pytorch, loss, deep-learning, neural-network]
---

# PyTorch Loss 函数详解

本文详细介绍 PyTorch 中常用的各类 Loss 函数，包括数学原理、代码实现和避坑指南。

## 一、回归任务 (Regression)

### 1. MSE Loss (均方误差)

**原理：** 计算预测值与真实值差值的平方均值。对误差大的点惩罚极重。

-   **数学公式：**
    $$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
    其中 $N$ 是样本数，$y_i$ 是真实值，$\hat{y}_i$ 是预测值。

-   **PyTorch 实现：**

    ```python
    import torch
    import torch.nn as nn

    # 假设 Batch Size = 2
    predictions = torch.tensor([2.5, 0.0], requires_grad=True)
    targets = torch.tensor([3.0, -0.5])

    criterion = nn.MSELoss()
    loss = criterion(predictions, targets)

    print(loss) # ((2.5-3.0)^2 + (0.0 - (-0.5))^2) / 2 = (0.25 + 0.25) / 2 = 0.25
    ```

### 2. MAE (L1) Loss (平均绝对误差)

**原理：** 计算差值的绝对值。梯度恒定，对异常值更包容。

-   **数学公式：**
    $$L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

-   **PyTorch 实现：**

    ```python
    criterion = nn.L1Loss()
    loss = criterion(predictions, targets)
    ```

### 3. Smooth L1 Loss (Huber Loss 的特例)

**原理：** 结合了 MSE 和 MAE。当误差 $|x| < \beta$ 时使用平方项（MSE），否则使用线性项（MAE）。这使得在 0 点处可导（Smooth），且对离群点不敏感。

-   **数学公式：**

    $$
    loss(x, y) = \frac{1}{N} \sum_{i} z_i
    $$

    $$
    z_i = \begin{cases}
    0.5 (y_i - \hat{y}_i)^2, & \text{if } |y_i - \hat{y}_i| < 1 \\
    |y_i - \hat{y}_i| - 0.5, & \text{otherwise}
    \end{cases}
    $$

-   **PyTorch 实现：**

    ```python
    # beta 默认为 1.0
    criterion = nn.SmoothL1Loss(beta=1.0)
    loss = criterion(predictions, targets)
    ```

---

## 二、分类任务 (Classification)

### 1. BCEWithLogitsLoss (二分类交叉熵)

**注意：** 在 PyTorch 中，强烈建议使用 `BCEWithLogitsLoss` 而不是 `BCELoss`。前者将 **Sigmoid** 层和 **BCELoss** 结合在一个类中，数值稳定性更高（Log-Sum-Exp 技巧）。

-   **数学公式：**
    $$L = - \frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(\sigma(x_i)) + (1 - y_i) \cdot \log(1 - \sigma(x_i))]$$
    其中 $x_i$ 是模型的原始输出 (Logits)，$\sigma(x)$ 是 Sigmoid 函数 $\frac{1}{1+e^{-x}}$。

-   **PyTorch 实现：**

    ```python
    # 二分类模型输出不需要加 Sigmoid，直接输出 Logits
    logits = torch.tensor([0.8, -0.5], requires_grad=True)
    targets = torch.tensor([1.0, 0.0]) # 标签必须是 float

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, targets)
    ```

### 2. CrossEntropyLoss (多分类交叉熵)

**注意：** PyTorch 的 `nn.CrossEntropyLoss` 已经**包含**了 Softmax 操作。**千万不要**在模型输出层再加 Softmax，否则相当于做了两次 Softmax，会导致模型难以训练。

-   **数学公式：**
    $$L = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{e^{x_{i, y_i}}}{\sum_{j} e^{x_{i, j}}} \right) = - \frac{1}{N} \sum_{i=1}^{N} x_{i, y_i} + \log(\sum_{j} e^{x_{i, j}})$$
    简单理解就是：$-\log(\text{真实类别对应的预测概率})$。

-   **PyTorch 实现：**

    ```python
    # 假设 3 分类，Batch Size = 2
    # 模型输出 Raw Logits (未经过 Softmax)
    logits = torch.tensor([[1.5, 0.2, -0.5], [0.1, 2.0, 0.5]], requires_grad=True)

    # 标签是类别的索引 (Long/Int)，不是 One-hot 编码
    targets = torch.tensor([0, 1])

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    ```

### 3. Focal Loss (自定义实现)

PyTorch 官方库暂未直接包含 Focal Loss，通常需要自己写一个类。

-   **数学公式：**
    $$L_{FL} = - \alpha (1 - p_t)^\gamma \log(p_t)$$
    其中 $p_t$ 是模型对真实类别的预测概率。$(1-p_t)^\gamma$ 是调节因子，当样本很难分（$p_t$ 小）时，权重变大。

-   **PyTorch 实现：**

    ```python
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.bce = nn.BCEWithLogitsLoss(reduction='none') # 不进行求和，保留每个样本的 loss

        def forward(self, inputs, targets):
            bce_loss = self.bce(inputs, targets)
            pt = torch.exp(-bce_loss) # 还原出 p_t
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()

    # 使用
    criterion = FocalLoss()
    loss = criterion(logits, targets)
    ```

---

## 三、分割与检测 (Segmentation & Detection)

### 1. Dice Loss

常用于医学图像分割。PyTorch 没有内置，需要手写。

-   **数学公式：**
    $$L_{Dice} = 1 - \frac{2 \sum (y_i \cdot \hat{y}_i) + \epsilon}{\sum y_i + \sum \hat{y}_i + \epsilon}$$
    其中 $\epsilon$ (smooth) 是为了防止分母为 0 的微小数值。

-   **PyTorch 实现：**

    ```python
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-5):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            # inputs 通常经过 Sigmoid 变为概率
            inputs = torch.sigmoid(inputs)

            # 展平张量，计算全局 Dice
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

            return 1 - dice

    # 使用
    criterion = DiceLoss()
    # 假设图片是 1x10x10
    seg_pred = torch.randn(1, 1, 10, 10, requires_grad=True)
    seg_target = torch.randint(0, 2, (1, 1, 10, 10)).float()
    loss = criterion(seg_pred, seg_target)
    ```

### 2. IoU / GIoU / CIoU Loss

在 `torchvision` (0.15+) 的 `ops` 模块中已经内置了相关函数。

-   **数学公式 (IoU Loss)：**
    $$Loss = 1 - IoU = 1 - \frac{Area(box1 \cap box2)}{Area(box1 \cup box2)}$$

-   **PyTorch 实现：**

    ```python
    from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss

    # 格式: [x1, y1, x2, y2]
    pred_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]], requires_grad=True)
    target_boxes = torch.tensor([[105.0, 105.0, 205.0, 205.0]])

    # CIoU Loss (YOLO常用)
    loss = complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')
    ```

---

## 四、生成与对比 (Generative & Contrastive)

### 1. KL Divergence Loss (KL 散度)

衡量两个分布的差异。

-   **数学公式：**
    $$D_{KL}(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)} = \sum P(x) (\log P(x) - \log Q(x))$$
    **注意：** PyTorch 的实现有点反直觉。

    -   输入 (Input/Pred): 应该是 **Log Probabilities** (log\_softmax 的结果)。
    -   目标 (Target/Label): 应该是 **Probabilities** (普通概率分布)。

-   **PyTorch 实现：**

    ```python
    # input 必须是 log_softmax 后的
    pred_log_probs = torch.nn.functional.log_softmax(torch.randn(2, 3), dim=1)
    # target 是概率分布
    target_probs = torch.softmax(torch.randn(2, 3), dim=1)

    criterion = nn.KLDivLoss(reduction='batchmean')
    loss = criterion(pred_log_probs, target_probs)
    ```

### 2. Triplet Margin Loss (三元组损失)

常用于人脸识别。拉近 Anchor 和 Positive，推开 Anchor 和 Negative。

-   **数学公式：**
    $$L = \max(d(a, p) - d(a, n) + margin, 0)$$
    其中 $d(x,y)$ 是欧氏距离。

-   **PyTorch 实现：**

    ```python
    anchor = torch.randn(10, 128, requires_grad=True)
    positive = torch.randn(10, 128, requires_grad=True)
    negative = torch.randn(10, 128, requires_grad=True)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    loss = criterion(anchor, positive, negative)
    ```

---

## 核心避坑指南

### 1. Logits vs Probabilities

-   用 `CrossEntropyLoss` 和 `BCEWithLogitsLoss` 时，模型**不要**加 Softmax/Sigmoid。
-   用 `MSELoss`、`DiceLoss` 时，通常需要激活函数（视具体值域而定）。

### 2. Reduction

-   大多数 Loss 默认 `reduction='mean'` (求均值)。
-   如果你想看每个样本的 Loss，设为 `reduction='none'`。

### 3. Type 错误

-   `CrossEntropyLoss` 的 target 通常是 `long` (int64) 类型。
-   `MSELoss`、`BCE` 的 target 必须是 `float` 类型。

### 4. 数值稳定性

-   始终优先使用 `BCEWithLogitsLoss` 而不是 `nn.Sigmoid()` + `nn.BCELoss()`。
-   使用 `CrossEntropyLoss` 而不是手动 `softmax` + `log` + `nll_loss`。

---

## 总结

选择合适的 Loss 函数是训练深度学习模型的关键：

| 任务类型 | 推荐 Loss | 注意事项 |
|---------|----------|---------|
| 回归 | MSELoss, L1Loss | 异常值多时用 L1 或 Smooth L1 |
| 二分类 | BCEWithLogitsLoss | 不要在模型加 Sigmoid |
| 多分类 | CrossEntropyLoss | 不要在模型加 Softmax |
| 类别不平衡 | Focal Loss | 自定义实现 |
| 图像分割 | Dice Loss, BCE+Dice | 医学图像常用 |
| 目标检测 | CIoU Loss | YOLO 系列常用 |
| 度量学习 | Triplet Loss | 人脸识别、ReID |
| 生成模型 | KL Divergence | VAE 中常用 |

:::tip
在实际应用中，经常会组合多个 Loss 函数，如分割任务中使用 `BCE + Dice Loss`，目标检测中使用 `分类 Loss + 回归 Loss + IoU Loss`。
:::
