---
title: Android集成flipper
sidebar_label: Android集成flipper
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android集成flipper

Flipper 是一个开源的平台，用于调试 iOS、Android 和 React Native 应用程序。它由 Facebook 开发和维护，提供了一个统一的界面，开发者可以通过各种插件来查看和调试应用的不同部分，比如网络请求、数据库、布局、日志等等。

Flipper 的主要功能

-   网络调试：查看和调试应用中的网络请求和响应，包括 HTTP 请求的详细信息。
-   数据库调试：查看和调试应用使用的 SQLite 数据库。
-   布局检查：查看和调试应用的视图层次结构，帮助识别布局问题。
-   日志查看：查看应用的日志输出，便于调试和分析问题。
-   性能监控：查看应用的性能指标，如内存使用情况、CPU 使用率等。
-   自定义插件：开发者可以创建自定义插件，以满足特定的调试需求。

## Android集成Flipper

为什么想要集成flipper，因为感觉flipper有一些很好用的插件，在调试和开发的时候很好用，尤其是flipper的网络监控功能，但是这两天发现现在Android studio现在也有网络请求inspect功能，所以反而网络请求抓取功能貌似也不太需要了。

[flipper文档](https://fbflipper.com/docs/tutorial/android/)

[android集成flipper demo](https://github.com/Hao-yiwen/android-study/tree/master/xml-and-compose-view-samples)

:::danger
目前在集成flipper时候使用0.182.0版本，最新版本在gradle集成的时候存在无法下载aar的问题。
:::
