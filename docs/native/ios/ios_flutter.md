---
title: iOS 原生工程集成 Flutter 模块（Add-to-App）
sidebar_label: iOS 集成 Flutter
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# iOS 原生工程集成 Flutter 模块（Add-to-App）

## 官方指南

- [Integrate a Flutter module into your iOS project](https://docs.flutter.dev/add-to-app/ios/project-setup)

## 集成与引擎初始化要点

- 按照官方指引通过 **CocoaPods** 将 Flutter 模块构建产物链接进现有 iOS 工程。
- **引擎预热（`FlutterEngine`）**：生产环境官方推荐在 App 启动或空闲时机提前预热 `FlutterEngine`，以消除首次打开页面的引擎冷启动延迟；即便在测试场景下仅于 `ViewController` 按需初始化引擎，其首屏可交互时间（TTI）表现依然优于冷加载 JS Bundle 的传统 RN 页面。

## 联调与断点调试步骤

1. 在 Xcode 中以 Debug 模式启动宿主 iOS 应用（确保进程处于可附加调试状态）。
2. 打开 VS Code 的 Flutter 模块工程，选择对应的运行模拟器或真机设备。
3. 执行 **`Flutter: Attach to Flutter on Device`（Dart Attach）**，即可对混合栈中的 Dart 代码进行热重载与断点调试。
