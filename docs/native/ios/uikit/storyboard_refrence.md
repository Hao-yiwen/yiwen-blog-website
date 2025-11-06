---
title: Storyboard References
sidebar_label: Storyboard References
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Storyboard References

Storyboard References 是 iOS 开发中用来在一个 Storyboard 中引用另一个 Storyboard 的功能。这种方法有助于将应用程序拆分为多个 Storyboard，从而简化管理、提高可维护性，并支持更好的团队协作。

## 为什么使用 Storyboard References

-   模块化管理：将应用程序拆分为多个 Storyboard，每个 Storyboard 负责不同的功能模块。
-   提高可维护性：减少单个 Storyboard 的复杂度，提高可读性和可维护性。
-   团队协作：多个开发者可以同时工作在不同的 Storyboard 上，减少冲突。
-   性能优化：加载更小的 Storyboard 可以提高应用的启动性能。

## 如何使用 Storyboard References

1. 创建多个 Storyboard

首先，创建多个 Storyboard 文件。例如，创建一个 Main.storyboard 和一个 Auth.storyboard。

2. 配置目标 Storyboard

在 Auth.storyboard 中配置一个视图控制器，并设置它的 Storyboard ID。

-   打开 Auth.storyboard。
-   添加视图控制器：从对象库中拖动一个 UIViewController 到 Storyboard 中。
-   设置 Storyboard ID：选择该视图控制器，在属性检查器中设置 Storyboard ID（例如，AuthViewController）。

3. 在源 Storyboard 中添加 Storyboard Reference

在 Main.storyboard 中添加一个 Storyboard Reference，指向 Auth.storyboard。

1. 打开 Main.storyboard。
2. 添加 Storyboard Reference：从对象库中拖动一个 Storyboard Reference 到需要引用的地方。
3. 设置引用的 Storyboard 名称和入口视图控制器的标识符：
    - Referenced Storyboard：设置为 Auth。
    - Storyboard ID：设置为 AuthViewController。
