---
title: Matplotlib 入门教程
sidebar_label: Matplotlib 入门
sidebar_position: 10
tags: [python, matplotlib, 数据可视化]
---

# Matplotlib 入门教程

Matplotlib 是 Python 中最基础、最流行的数据可视化库。几乎所有你在 Python 中见到的静态图表，底层逻辑大概率都和它有关（甚至很多更高级的库如 Seaborn、Pandas Plotting 都是基于它构建的）。

## 1. `plt` 到底是什么？

在几乎所有的 Matplotlib 教程中，你都会看到这行代码：

```python
import matplotlib.pyplot as plt
```

这里的 **`plt`** 只是一个通用的缩写（别名），它指代的是 Matplotlib 库中的 **`pyplot`** 子模块。

### 为什么它这么重要？

- **它是入口：** `pyplot` 提供了一个"状态机式"的接口。简单来说，它就像一个**遥控器**。
- **它是画笔：** 当你调用 `plt.plot()` 时，你是在告诉这个遥控器："在当前活动的画布上，用当前的画笔画一条线"。
- **它是为了模仿 MATLAB：** Matplotlib 的设计初衷之一是让习惯使用 MATLAB 的工程师能快速上手 Python 画图，`pyplot` 的语法风格几乎照搬了 MATLAB。

> **一句话总结：** `plt` 是你与 Matplotlib 交互的最便捷接口，负责管理画布、坐标轴和绘图动作。

## 2. 核心概念：Matplotlib 的层级结构

初学者最容易晕的地方在于分不清"画布"和"坐标系"。你可以把 Matplotlib 的绘图逻辑想象成**在墙上挂画**：

### Figure (画布/图窗)

这是最底层的容器，相当于**整张纸**或**整个窗口**。一张 `Figure` 上可以包含一个或多个图表。

### Axes (子图/绘图区)

**这是最重要的概念！** 注意不是 Axis（轴），而是 Axes。

它代表**具体的某一张图表**。它包含坐标轴、曲线、标题等。如果你想在一个窗口画两张图，那你就有 1 个 `Figure` 和 2 个 `Axes`。

### Axis (坐标轴)

仅仅指 X 轴和 Y 轴这两条线，以及上面的刻度（Ticks）和标签（Labels）。

### Artist (图元)

你在图上看到的任何东西（点、线、字、图例）都是 Artist。

## 3. 两种使用流派

Matplotlib 有两种写代码的方式，初学者常因为混淆这两种方式而感到困惑。

### 方式 A：Pyplot 风格 (简单、快捷)

这是最常见的"脚本式"写法，依靠 `plt` 自动管理当前状态。适合简单的图表。

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])  # 只要这一句就能画图
plt.title("Simple Plot")
plt.show()
```

### 方式 B：面向对象风格 (推荐)

这是更专业、更灵活的写法。你需要显式地创建"画布"和"子图"对象，然后对着对象操作。**它可以让你精确控制图表的每一个细节。**

```python
import matplotlib.pyplot as plt

# 1. 显式创建 画布(fig) 和 子图(ax)
fig, ax = plt.subplots()

# 2. 在子图对象上画图
ax.plot([1, 2, 3], [4, 5, 6])

# 3. 设置子图属性 (注意这里用 set_title 而不是 title)
ax.set_title("OO Style Plot")

plt.show()
```

> **建议：** 虽然 `plt.plot()` 很简单，但为了长远考虑（比如画多子图、复杂布局），**强烈建议养成使用 `fig, ax = plt.subplots()` 的习惯**。

## 4. 常用图表速查

使用 `plt` (或 `ax` 对象) 可以绘制多种图表，以下是最高频的几种：

| 图表类型   | 方法名           | 适用场景                         |
| :--------- | :--------------- | :------------------------------- |
| **折线图** | `plot(x, y)`     | 观察数据随时间或顺序的变化趋势   |
| **散点图** | `scatter(x, y)`  | 观察两个变量之间的相关性         |
| **柱状图** | `bar(x, height)` | 比较不同类别的数值大小           |
| **直方图** | `hist(x)`        | 查看数据的分布情况               |
| **饼图**   | `pie(x)`         | 查看各部分占总体的比例           |

## 5. 实战代码示例

这是一个完整的、带有注释的面向对象风格示例：

```python
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
x = np.linspace(0, 10, 100)  # 0到10之间生成100个点
y1 = np.sin(x)               # 正弦曲线
y2 = np.cos(x)               # 余弦曲线

# 2. 创建画布和子图
# figsize控制图片大小(宽, 高)
fig, ax = plt.subplots(figsize=(8, 5))

# 3. 绘图并设置样式
# label用于后续显示图例
ax.plot(x, y1, color="blue", linewidth=2, linestyle="-", label="Sin(x)")
ax.plot(x, y2, color="red",  linewidth=2, linestyle="--", label="Cos(x)")

# 4. 装饰图表
ax.set_title("Trigonometric Functions", fontsize=14) # 标题
ax.set_xlabel("Time (s)")                            # X轴标签
ax.set_ylabel("Amplitude")                           # Y轴标签
ax.grid(True, alpha=0.3)                             # 显示网格，透明度0.3
ax.legend()                                          # 显示图例

# 5. 显示或保存
# plt.savefig("my_plot.png", dpi=300) # 保存为图片
plt.show()                            # 在屏幕显示
```

## 6. 常用样式参数

### 颜色

```python
color='blue'      # 颜色名
color='#FF5733'   # 十六进制
color='r'         # 简写：r红, g绿, b蓝, k黑, w白
```

### 线型

```python
linestyle='-'     # 实线
linestyle='--'    # 虚线
linestyle='-.'    # 点划线
linestyle=':'     # 点线
```

### 标记点

```python
marker='o'        # 圆点
marker='s'        # 方块
marker='^'        # 三角形
marker='*'        # 星号
```

## 7. 多子图布局

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

# 创建 2行2列 的子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 访问各个子图
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Cos')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('Tan')

axes[1, 1].plot(x, x**2)
axes[1, 1].set_title('Square')

plt.tight_layout()  # 自动调整布局，避免重叠
plt.show()
```

## 8. 保存图片

```python
# 保存为不同格式
plt.savefig('plot.png', dpi=300)           # PNG，高分辨率
plt.savefig('plot.pdf')                     # PDF，矢量图
plt.savefig('plot.svg')                     # SVG，矢量图

# 常用参数
plt.savefig('plot.png',
            dpi=300,                        # 分辨率
            bbox_inches='tight',            # 去除多余空白
            transparent=True)               # 透明背景
```

## 相关资源

- [Matplotlib 官方文档](https://matplotlib.org/stable/contents.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - 官方示例库
