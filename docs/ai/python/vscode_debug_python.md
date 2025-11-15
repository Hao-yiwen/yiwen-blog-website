---
title: VS Code 中调试 Python 文件的简单指南
sidebar_position: 10
tags: [python, vscode, 调试, debug]
---

# 📘 VS Code 中调试 Python（.py）文件的简单指南

本文介绍如何在 **Visual Studio Code** 中"丝滑地"调试 Python 文件，包括环境选择、断点调试、配置启动项等基础操作。

---

## 1. 安装必要插件

在 VS Code 左侧 **Extensions（扩展）** 中安装：

* **Python**（Microsoft 官方）
* **Pylance**（智能补全与类型推断）
* **Python Debugger**（调试支持）

安装完成后建议 **重启 VS Code**。

---

## 2. 选择 Python 解释器（非常重要）

VS Code 在不同项目中可能有多个环境（如 venv、conda、系统 python）。

选择解释器：

1. 打开任意 `.py` 文件
2. 使用快捷键：

   * macOS：`Cmd + Shift + P`
   * Windows / Linux：`Ctrl + Shift + P`
3. 输入：`Python: Select Interpreter`
4. 选择你项目使用的 Python 环境（如 `.venv/bin/python`）

> 正确选择解释器可以避免"断点不生效""调试环境不一致"等问题。

---

## 3. 开始调试（F5）

打开你要调试的 `.py` 文件，直接按 **F5**：

* VS Code 会启动调试模式
* 代码运行会在断点处暂停
* 左侧会显示变量、调用栈、监视（Watch）等调试信息

第一次调试时，VS Code 会自动生成 `.vscode/launch.json` 文件。

---

## 4. 添加断点

点击代码行左侧的空白即可添加断点（红点）。

支持：

* **普通断点**
* **条件断点**
  右键断点 → Add Conditional Breakpoint
  例：`i > 100`

---

## 5. 常用调试按键

| 按键          | 功能              |
| ----------- | --------------- |
| F5          | 开始 / 继续执行       |
| F10         | 单步跳过（Step Over） |
| F11         | 进入函数（Step Into） |
| Shift + F11 | 跳出函数（Step Out）  |
| Shift + F5  | 停止调试            |

---

## 6. 查看变量

调试暂停时，左侧会自动显示：

* **本地变量**
* **全局变量**
* **自定义监视变量（Watch）**

鼠标悬停在变量上也可直接查看其值。

---

## 7. 调试控制台（Debug Console）

在暂停时，你可以在 Debug Console 中直接运行 Python 表达式：

```python
a
a + 10
len(my_list)
```

这是调试中最强大的功能之一，可用于快速验证变量状态。

---

## 8. 自定义调试配置（可选）

如需更灵活的调试配置，在项目根目录创建：

`.vscode/launch.json`

推荐的基础配置如下：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Python",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

常用扩展字段：

* 添加命令行参数：

```json
"args": ["--epochs", "50"]
```

* 设置环境变量：

```json
"env": {
  "PYTHONUNBUFFERED": "1"
}
```

* 指定入口文件：

```json
"program": "${workspaceFolder}/main.py"
```

---

## 9. Tips：更丝滑的体验

* 打开 `.py` 文件时 VS Code 会自动识别 Python
* 使用虚拟环境可保持依赖干净
* 在调试 GPU（如 PyTorch）时可插入：

  ```python
  torch.cuda.synchronize()
  ```

  避免步进乱跳
