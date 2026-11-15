---
title: Flutter 隐式动画（Implicit Animations）
sidebar_label: 隐式动画
date: 2024-08-08
last_update:
  date: 2024-08-08
---

# Flutter 隐式动画（Implicit Animations）

隐式动画组件无需手动管理 `AnimationController`，只要目标属性发生变化，框架就会在指定 `duration` 内自动插值过渡。

## 常用隐式动画组件

### 1. `AnimatedContainer`

当宽高、颜色、边距、圆角等容器属性变化时自动平滑过渡：

```dart
AnimatedContainer(
    duration: const Duration(seconds: 1),
    color: Colors.blue,
    width: 200,
    height: 200,
)
```

### 2. `AnimatedOpacity`

当透明度变化时自动执行淡入淡出效果：

```dart
AnimatedOpacity(
    opacity: 0.5,
    duration: const Duration(seconds: 1),
    child: const Text('Hello World'),
)
```

### 3. `AnimatedAlign`

当对齐方位变化时自动平移动画：

```dart
AnimatedAlign(
    alignment: Alignment.topLeft,
    duration: const Duration(seconds: 1),
    child: const Text('Aligned Text'),
)
```

### 4. `AnimatedPadding`

当内边距变化时平滑缩放间距：

```dart
AnimatedPadding(
    padding: const EdgeInsets.all(20),
    duration: const Duration(seconds: 1),
    child: const Text('Padded Text'),
)
```

### 5. `AnimatedPositioned`

配合 `Stack` 布局使用，当绝对定位坐标或尺寸变化时自动过渡：

```dart
Stack(
    children: [
        AnimatedPositioned(
            top: 50,
            left: 50,
            duration: const Duration(seconds: 1),
            child: Container(
                width: 100,
                height: 100,
                color: Colors.blue,
            ),
        ),
    ],
)
```

### 6. `AnimatedSwitcher`

当子组件切换时（注意通过 `key` 区分新旧组件）自动执行过渡动画：

```dart
AnimatedSwitcher(
    duration: const Duration(seconds: 1),
    child: Text(
        'Animated Text',
        key: const ValueKey<int>(1),
    ),
)
```

### 7. `AnimatedCrossFade`

在两个固定子组件之间平滑交叉淡入淡出：

```dart
AnimatedCrossFade(
    duration: const Duration(seconds: 1),
    firstChild: const Text('First'),
    secondChild: const Text('Second'),
    crossFadeState: CrossFadeState.showFirst,
)
```

### 8. `AnimatedDefaultTextStyle`

当字号、颜色、字重等文本样式变化时自动插值过渡：

```dart
AnimatedDefaultTextStyle(
    style: const TextStyle(fontSize: 24, color: Colors.blue),
    duration: const Duration(seconds: 1),
    child: const Text('Animated Text Style'),
)
```
