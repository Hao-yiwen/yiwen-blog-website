---
title: viewController常用的生命周期方法
sidebar_label: viewController常用的生命周期方法
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import ios_viewcontroller_lifecycle from '@site/static/img/ios_viewcontroller_lifecycle.png'

# viewController常用的生命周期方法

UIViewController 是 iOS 开发中用于管理视图层次结构的基础类。它有一系列生命周期函数，这些函数允许你在不同的阶段执行特定的代码。了解这些生命周期函数非常重要，因为它们决定了视图控制器在应用程序运行时的行为。

## android生命周期方法

[android生命周期函数文档](../../android/life_cycle.md)

## UIViewController 生命周期函数

<img src={ios_viewcontroller_lifecycle} width={300} />

1. loadView:
    - 调用时机：当视图控制器的视图属性被访问，但视图尚未加载时调用。你可以在这里创建自定义的视图层次结构。
    - 官方文档：loadView

```swift
override func loadView() {
    super.loadView()
    // 自定义视图层次结构的代码
}
```

2. viewDidLoad:
    - 调用时机：当视图控制器的视图首次加载到内存时调用。
    - 用途：用于初始化视图、设置数据、添加子视图等。
    - 官方文档：viewDidLoad

```swift
override func viewDidLoad() {
    super.viewDidLoad()
    // 初始化代码
}
```

3. viewWillAppear:
    - 调用时机：每次视图控制器的视图即将加入视图层次结构并显示在屏幕上时调用。
    - 用途：用于更新视图的内容，例如从数据源重新加载数据。
    - 官方文档：viewWillAppear

```swift
override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    // 视图即将出现时的代码
}
```

4. viewDidAppear:
    - 调用时机：每次视图控制器的视图已经加入视图层次结构并显示在屏幕上时调用。
    - 用途：用于开始动画、开始网络请求、启动定时器等。
    - 官方文档：viewDidAppear

```swift
override func viewDidAppear(_ animated: Bool) {
    super.viewDidAppear(animated)
    // 视图已经出现时的代码
}
```

5. viewWillDisappear:
    - 调用时机：每次视图控制器的视图即将从视图层次结构中移除时调用。
    - 用途：用于保存数据、取消网络请求、停止动画等。
    - 官方文档：viewWillDisappear

```swift
override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    // 视图即将消失时的代码
}
```

6. viewDidDisappear:
    - 调用时机：每次视图控制器的视图已经从视图层次结构中移除时调用。
    - 用途：用于释放资源、停止服务、停止定时器等。
    - 官方文档：viewDidDisappear

```swift
override func viewDidDisappear(_ animated: Bool) {
    super.viewDidDisappear(animated)
    // 视图已经消失时的代码
}
```

7. viewWillLayoutSubviews:
    - 调用时机：当视图控制器的视图即将布局其子视图时调用。
    - 用途：用于更新子视图的布局。
    - 官方文档：viewWillLayoutSubviews

```swift
override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    // 子视图即将布局时的代码
}
```

8. viewDidLayoutSubviews:
    - 调用时机：当视图控制器的视图已经布局其子视图时调用。
    - 用途：用于执行在子视图布局完成后的任何操作。
    - 官方文档：viewDidLayoutSubviews

```swift
override func viewDidLayoutSubviews() {
    super.viewDidLayoutSubviews()
    // 子视图已经布局时的代码
}
```

9. didReceiveMemoryWarning:
    - 调用时机：当应用收到内存警告时调用。
    - 用途：用于释放不必要的资源，以减少内存占用。
    - 官方文档：didReceiveMemoryWarning

```swift
override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // 处理内存警告的代码
}
```
