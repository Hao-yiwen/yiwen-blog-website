---
title: Expo
sidebar_label: Expo
date: 2024-08-31
last_update:
  date: 2024-08-31
---

# Expo

## 文档

https://docs.expo.dev/

## 介绍

在React Native@074版本开始，官方开始推荐Expo作为RN首选框架，Expo基于RN做了深度定制。推出了托管工作流，Expo SDK，并且还有EAS支持构建，极大简化了开发流程，以下是对特点的介绍。后续RN重点将会在Expo方面，因为Expo的SDK和托管工作流极大的简化了开发。当然最大的好处就是我不用每次需要再本地构建，只需要使用Orbit从云端下载构建就可以。因为RN本地构建真的太恶心了。

## 版本对应关系

https://docs.expo.dev/versions/latest/

### 托管工作流（Managed Workflow）

-   快速启动：Expo 提供了一个托管工作流，开发者无需配置复杂的原生开发环境即可开始构建 React Native 应用。只需安装 Expo CLI，便可以创建和运行一个 React Native 应用。
-   内置功能：托管工作流包括了许多常见的移动应用功能，如相机访问、位置服务、推送通知、媒体处理等，这些功能可以通过 Expo SDK 中的模块直接使用，无需额外的配置。
-   自动更新：使用托管工作流构建的应用可以通过 Expo 的云服务自动接收更新，无需重新提交到应用商店。这对于快速迭代应用非常有用。

### 裸工作流（Bare Workflow）

-   完全控制：裸工作流允许开发者完全控制 iOS 和 Android 的原生项目，适合需要自定义原生代码的项目。开发者可以自由使用任何 React Native 库，并且仍然可以使用 Expo 的部分服务（如 EAS 构建和更新）。
-   兼容性：裸工作流在允许自定义原生代码的同时，仍然兼容大部分 Expo SDK 提供的功能，给开发者提供了灵活性和便利性。

### Expo SDK

-   丰富的模块：Expo SDK 提供了大量预构建的功能模块，这些模块覆盖了移动应用开发的常见需求，如相机、传感器、位置、媒体、文件系统等。开发者可以通过简单的 API 调用来实现复杂的功能，而不需要编写任何原生代码。
-   跨平台支持：Expo SDK 的所有模块都支持 iOS 和 Android 平台，开发者只需编写一次代码，即可在两个平台上运行。

### Expo Application Services (EAS)

-   EAS Build：EAS Build 是一个云端构建服务，支持构建 iOS 和 Android 应用包。它简化了构建过程，开发者无需本地设置复杂的构建环境即可在云端完成应用的构建。
-   EAS Submit：EAS Submit 允许开发者将构建好的应用包直接提交到 App Store 和 Google Play。这减少了提交过程中的手动操作，并加快了发布周期。
-   EAS Update：EAS Update 允许开发者在不提交新版本的情况下，直接推送 JavaScript 和资源更新到用户的设备。这对于快速修复 bug 或更新内容非常有用。
