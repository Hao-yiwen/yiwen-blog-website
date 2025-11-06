---
title: development pod和pod
sidebar_label: development pod和pod
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# development pod和pod

## 使用cocopod添加依赖,为什么有些依赖在development pod中还有一些在pod中?

在使用 CocoaPods 添加依赖时，有些依赖被放在 Development Pods 中，而另一些则在 Pods 中。这种区分的原因如下：

1. Development Pods：

-   这些 Pods 通常是你的项目中正在开发的模块或组件。
-   你希望直接在你的工作空间中修改和调试这些 Pods。
-   它们通常是使用本地路径或 :path 选项在 Podfile 中指定的。

2. Pods：

-   这些是外部库或已经稳定的内部库。
-   它们通常来自于 CocoaPods 仓库（例如，CocoaPods 主库或自定义的私有库）。
-   这些 Pods 是你项目的依赖项，但你通常不会直接修改它们。

## 示例

你的 Podfile 可能如下所示：

```ruby
# Development Pods
pod 'MyLocalPod', :path => '../MyLocalPod'

# External Pods
pod 'AFNetworking', '~> 4.0'
```
