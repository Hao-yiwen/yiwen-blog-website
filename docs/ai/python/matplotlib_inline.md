---
title: matplotlib inline 魔术命令详解
sidebar_label: "%matplotlib inline"
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# %matplotlib inline

`%matplotlib inline` 是 **Jupyter Notebook 的魔术命令**（magic command），用于设置 matplotlib 图表的显示方式。

## 作用

- 让 matplotlib 生成的图表**直接显示在 Notebook 的单元格下方**
- 图表会嵌入到 Notebook 中,而不是弹出新窗口

## 示例

```python
%matplotlib inline
import matplotlib.pyplot as plt

# 画图后会直接在单元格下方显示
plt.plot([1, 2, 3, 4])
plt.show()
```

## 其他相关命令

- `%matplotlib notebook` - 生成交互式图表（可以缩放、平移）
- `%matplotlib widget` - 更现代的交互式图表
- 不加这个命令 - 图表可能不显示或弹出新窗口

## 注意事项

- 这个命令只需要在 Notebook 开头运行**一次**
- 只在 Jupyter Notebook/JupyterLab 中有效
- 在普通 Python 脚本中不需要也无效

## 补充说明

现在很多 Jupyter 环境已经默认启用了这个功能,所以有时候即使不写这行代码图表也能正常显示。
