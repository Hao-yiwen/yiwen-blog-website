---
title: ios集成flutter
sidebar_label: ios集成flutter
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# ios集成flutter

## 文档

[文档](https://docs.flutter.dev/add-to-app/ios/project-setup)

## 说明

-   按文档进行操作，使用cocopods将产物打进拿到ios构建中。

-   flutter官方推荐flutter引擎预热，而不是在用到的时候进行初始化，但是因为是测试页面，所以这里使用在viewcontroller用到的时候才会初始化引擎。页面tti依然haoyuan直接达成jsbudnle的RN页面。

## flutter调试

-   在vscode选择要调试的设备

-   在vscode启动`Dart attach`

-   ios设备需要进行debug process
