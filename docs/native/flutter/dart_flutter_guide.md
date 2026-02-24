---
sidebar_position: 2
title: Dart与Flutter核心语法指南
tags: [dart, flutter, guide]
---

# Dart与Flutter核心语法指南

这份指南为你详细梳理了学习 Flutter 开发最核心的 Dart 语法、Flutter 基础架构概念以及最常用的 UI 组件。

## 一、Dart 常用语法详解

Dart 是 Flutter 的开发语言，掌握以下核心语法是编写 Flutter 应用的基础。

### 1. 变量声明与数据类型

Dart 是一门强类型语言，但也支持类型推断。

- **`var`**: 自动推断类型，一旦推断确定，不能更改为其他类型。
- **`dynamic`**: 动态类型，编译时不检查类型，运行时可以赋任何类型的值（尽量少用）。
- **`final` vs `const`**: 都用于声明常量。`final` 是运行时常量（如获取当前时间），`const` 是编译时常量（如固定的数字或字符串）。

```dart
var name = 'Gemini'; // 推断为 String
int age = 3;
double height = 1.75;
bool isAI = true;

final currentTime = DateTime.now(); // 只能用 final，运行时才确定
const pi = 3.14159;                 // 编译时就确定的常量
```

### 2. 空安全 (Null Safety)

现代 Dart 强制使用空安全，防止由于尝试访问 null 对象而导致的崩溃。

- **`?`**: 表示变量可以为 null。
- **`??`**: 如果左侧不为 null 则返回左侧，否则返回右侧（提供默认值）。
- **`!`**: 非空断言，明确告诉编译器"我保证这个变量现在绝对不是 null"。

```dart
String? nullableName; // 可以为 null
String defaultName = nullableName ?? '默认名字'; // 如果为null，则使用'默认名字'
int length = nullableName!.length; // 危险操作：如果 nullableName 为 null，程序会崩溃
```

### 3. 函数 (Functions)

Dart 中函数也是对象。支持箭头语法和可选参数。

```dart
// 常规函数
int add(int a, int b) {
  return a + b;
}

// 箭头函数 (只有一行代码时使用)
int multiply(int a, int b) => a * b;

// 命名可选参数 (Flutter组件中极其常用)
// 使用 {} 包装参数，并使用 required 关键字要求必传
void printInfo({required String name, int? age, String city = '北京'}) {
  print('$name is $age from $city');
}
```

### 4. 异步编程 (Async / Await)

Flutter 中涉及网络请求、文件读取等耗时操作时，必须使用异步。

- **`Future`**: 代表一个未来的值。
- **`async` / `await`**: 让异步代码看起来像同步代码。

```dart
Future<String> fetchUserData() async {
  // 模拟网络请求延迟 2 秒
  await Future.delayed(Duration(seconds: 2));
  return '用户数据加载成功';
}

void main() async {
  print('开始加载...');
  String data = await fetchUserData();
  print(data);
}
```

### 5. 面向对象 (类与构造函数)

Dart 完全支持面向对象，Flutter 所有的组件（Widget）都是类。

```dart
class Person {
  String name;
  int age;

  // 语法糖构造函数 (非常简洁)
  Person(this.name, this.age);

  // 命名构造函数
  Person.guest() : name = '访客', age = 0;
}
```

## 二、Flutter 常用语法与核心概念

Flutter 的核心思想是 **"一切皆组件 (Everything is a Widget)"**。无论是按钮、文字，还是边距、对齐方式，全都是 Widget。

### 1. StatelessWidget (无状态组件)

当 UI 不需要根据数据的变化而重新渲染时使用。比如纯展示的静态文本、图标。

```dart
import 'package:flutter/material.dart';

class MyTextWidget extends StatelessWidget {
  final String title;

  const MyTextWidget({Key? key, required this.title}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 每次渲染都会调用 build 方法
    return Text(title);
  }
}
```

### 2. StatefulWidget (有状态组件)

当 UI 需要根据用户操作、网络请求等发生动态变化时使用。比如计数器、复选框、表单。

它由两个类组成：`StatefulWidget` 类本身，以及对应的 `State` 类。

```dart
class CounterWidget extends StatefulWidget {
  @override
  _CounterWidgetState createState() => _CounterWidgetState();
}

class _CounterWidgetState extends State<CounterWidget> {
  int _count = 0;

  void _increment() {
    // 极其重要：setState 通知 Flutter 数据已更改，需要重新运行 build() 刷新屏幕
    setState(() {
      _count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: _increment,
      child: Text('点击次数: $_count'),
    );
  }
}
```

## 三、Flutter 常用组件 (Widgets)

可以按功能将这些组件分为四大类：

### 1. 基础结构组件 (Material Design)

搭建 App 骨架的必备组件。

- **`MaterialApp`**: App 的入口，提供主题、路由等全局配置。
- **`Scaffold`**: 页面脚手架，提供了顶部导航栏 (`AppBar`)、主体内容区 (`body`)、侧滑菜单 (`Drawer`) 和底部导航 (`BottomNavigationBar`) 等。
- **`AppBar`**: 顶部的应用导航栏。

### 2. 内容展示组件

- **`Text`**: 显示文本。支持 `TextStyle` 调整颜色、大小、粗细。
- **`Image`**: 显示图片。常用 `Image.network()` 加载网络图片，`Image.asset()` 加载本地资源图片。
- **`Icon`**: 显示矢量图标。如 `Icon(Icons.home, color: Colors.blue)`。
- **按钮类**:
    - `ElevatedButton`: 带有阴影的凸起按钮（最常用）。
    - `TextButton`: 扁平文字按钮，没有背景和阴影。
    - `IconButton`: 图标按钮。

### 3. 布局组件 (Layout)

用于控制子组件在屏幕上的排列方式。

- **`Container`**: 最常用的"盒子"。相当于前端的 `div`。可以设置宽高、背景色、边框、圆角（通过 `decoration`）、内边距 (`padding`) 和外边距 (`margin`)。
- **`Row`**: 水平方向排列子组件。类似于 Flexbox 的 `flex-direction: row`。
- **`Column`**: 垂直方向排列子组件。极其常用！
- **`Stack`**: 堆叠布局，允许子组件像图层一样相互覆盖（结合 `Positioned` 定位）。
- **`Expanded`**: 必须包裹在 `Row` 或 `Column` 中使用，用于占满剩余的可用空间。

### 4. 滚动与列表组件

当内容超出屏幕时必须使用，否则会报错（出现黄色黑色的警告条纹）。

- **`SingleChildScrollView`**: 让单一子组件可以滚动。通常包裹在一个 `Column` 外面。
- **`ListView`**: 用于显示长列表。如果列表项非常多，推荐使用 `ListView.builder()`，它具有懒加载机制，只渲染屏幕可见的项，性能极佳。
- **`GridView`**: 网格布局，常用于商品展示、相册等。

## 四、完整页面示例

综合以上知识，一个标准的 Flutter 页面代码如下：

```dart
import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      home: Scaffold(
        appBar: AppBar(title: const Text('我的 Flutter 页面')),
        body: Center( // 居中布局
          child: Column( // 垂直排布
            mainAxisAlignment: MainAxisAlignment.center, // 垂直居中
            children: [
              const Icon(Icons.star, size: 50, color: Colors.orange),
              const SizedBox(height: 20), // 用一个透明盒子制造间距
              const Text('欢迎学习 Flutter', style: TextStyle(fontSize: 24)),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  print('按钮被点击了!');
                },
                child: const Text('点击我'),
              )
            ],
          ),
        ),
      ),
    );
  }
}
```
