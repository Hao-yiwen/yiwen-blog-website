---
title: Android中的主题颜色定义
sidebar_label: Android中的主题颜色定义
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android中的主题颜色定义

## 文档

[material主题颜色自定义](https://m3.material.io/theme-builder#/custom)

## 操作步骤

1. 现在`material`中自定义颜色主题，然后获取到`theme.kt`和`color.kt`,将其中颜色部分复制到自己的项目中
2. 然后重构项目，就会发现主题颜色已经变成了自定义的主题颜色。

## 动态颜色

Material 3 非常注重用户体验，其中推出的动态配色这项新功能就能根据用户的壁纸为应用创建主题。这样一来，如果用户喜欢绿色且拥有蓝色的手机背景，Woof 应用也会据此呈现蓝色。动态主题仅适用于搭载 Android 12 及更高版本的特定设备。

```kt
@Composable
fun WoofTheme(
   darkTheme: Boolean = isSystemInDarkTheme(),
   dynamicColor: Boolean = true,
   content: @Composable () -> Unit
)
```