---
title: rn自动连接
sidebar_label: rn自动连接
date: 2024-07-10
last_update:
  date: 2024-07-10
---

# rn自动连接

一直对rn自动连接很好奇，在想到底如何做到的自动链接。今天有机会研究RN，所以对自动连接研究了一下。以android为例，自动连接实际上就是`@react-native-communtity/cli-platform-android/native_modules.gralde`脚本

[代码链接](https://github.com/react-native-community/cli/blob/main/packages/cli-platform-android/native_modules.gradle)

## 原理

-   手动连接的流程是先在gradle中添加依赖，然后在packagelist中初始化依赖。

-   而自动连接则是用gradle脚本自动化了这个过程。实际上工作还是一样的，只是自动化了。而且packagelist的实现绑定了application。所以如果改造rn项目，自动链接可能失效。

-   依赖于`npx react-native config`获取依赖信息。

-   而依赖信息则是由社区维护的脚手架通过**遍历rn项目**的依赖获取。

## 生效入口

`settings.gradle`中的

```java
apply from: file("../node_modules/@react-native-community/cli-platform-android/native_modules.gradle"); applyNativeModulesSettingsGradle(settings)
```
