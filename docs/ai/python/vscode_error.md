---
title: VSCode Notebook 导入报错解决指南
sidebar_label: VSCode Notebook 导入报错解决指南
date: 2025-11-04
last_update:
  date: 2025-11-04
---

# VSCode Notebook 导入报错解决指南

## 问题描述

在 VSCode 的 Jupyter Notebook 中，出现类似以下错误提示：

```
无法解析导入 "matplotlib.pyplot"
basedpyright__reportMissingImports
```

**症状：**
- ✅ 代码可以正常运行
- ❌ VSCode 显示红色波浪线
- ❌ IDE 提示找不到模块

## 原因分析

这是 **Pylance/Pyright 类型检查器** 的问题，原因是：

- **运行时环境**：Python 能找到包（所以代码能运行）
- **IDE 静态分析**：VSCode 使用的 Python 解释器找不到包

常见原因：
1. VSCode 选择的 Python 解释器不对
2. 包没有安装在当前环境
3. 虚拟环境配置问题

---

## 解决方案

**步骤：**

1. 按 `Ctrl + Shift + P` (Mac: `Cmd + Shift + P`)
2. 输入 `Python: Select Interpreter`
3. 选择安装了 matplotlib 的 Python 环境

**如何判断选对了？**

在 notebook 中运行：
```python
import sys
print("当前 Python 路径:", sys.executable)
```

确保输出的路径是你安装包的环境。

## 总结

**最常见原因：** VSCode 选择的 Python 解释器不对

**最快解决方法：** 
1. `Ctrl + Shift + P` → `Python: Select Interpreter`
2. 选择正确的环境
3. 重启语言服务器

**最佳实践：**
- 使用虚拟环境
- 配置项目默认解释器
- 维护 requirements.txt