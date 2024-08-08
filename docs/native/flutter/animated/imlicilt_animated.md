~~# 隐式动画

## 常用隐式动画组件

1. AnimatedContainer

-   当其属性（如宽度、高度、颜色、边框等）发生变化时，会自动执行动画。
-   示例：

```dart
AnimatedContainer(
duration: Duration(seconds: 1),
color: Colors.blue,
width: 200,
height: 200,
)
```

2. AnimatedOpacity

-   当透明度发生变化时，会自动执行淡入淡出的动画。
-   示例：

```dart
AnimatedOpacity(
opacity: 0.5,
duration: Duration(seconds: 1),
child: Text('Hello World'),
)
```

3. AnimatedAlign

-   当对齐方式发生变化时，会自动执行位置变化的动画。
-   示例：

```dart
AnimatedAlign(
alignment: Alignment.topLeft,
duration: Duration(seconds: 1),
child: Text('Aligned Text'),
)
```

4. AnimatedPadding

-   当填充（padding）发生变化时，会自动执行内边距变化的动画。
-   示例：

```dart
AnimatedPadding(
padding: EdgeInsets.all(20),
duration: Duration(seconds: 1),
child: Text('Padded Text'),
)
```

5. AnimatedPositioned

-   适用于 Stack 布局中，当定位位置发生变化时，会自动执行位置变化的动画。
-   示例：

```dart
Stack(
children: [
AnimatedPositioned(
    top: 50,
    left: 50,
    duration: Duration(seconds: 1),
    child: Container(
    width: 100,
    height: 100,
    color: Colors.blue,
    ),
),
],
)
```

6. AnimatedSwitcher

-   在子组件发生变化时，会自动执行子组件替换的动画。
-   示例：

```dart
AnimatedSwitcher(
duration: Duration(seconds: 1),
child: Text(
'Animated Text',
key: ValueKey<int>(1), // 每次改变需要不同的key
),
)
```

7. AnimatedCrossFade

-   在两个子组件之间交替显示时，会自动执行淡入淡出的动画。
-   示例：

```dart
AnimatedCrossFade(
duration: Duration(seconds: 1),
firstChild: Text('First'),
secondChild: Text('Second'),
crossFadeState: CrossFadeState.showFirst, // 或者 CrossFadeState.showSecond
)
```

8. AnimatedDefaultTextStyle

-   当文本样式发生变化时，会自动执行样式变化的动画。
-   示例：

```dart
AnimatedDefaultTextStyle(
style: TextStyle(fontSize: 24, color: Colors.blue),
duration: Duration(seconds: 1),
child: Text('Animated Text Style'),
)
```
