---
sidebar_position: 1
---

# ReactNative开发

## 发展历史

截至20241029，rn已经推出了稳定的新架构版本了，而此时rn的开发也完全和之前的开发逻辑不相同了。

- react native只维护rn的核心逻辑以及架构逻辑，但是在实际的业务开发中无法只使用rn库来进行app开发，需要大量依赖社区维护的库，所以rn团队觉得rn只是一个库，而不是一个开发框架，如果要进行app开发，那么推荐使用expo框架进行。
- 新架构在076已经默认使用，所以后续的开发要使用新架构开发。经过简单测试新架构+hbc的性能要远好于旧架构，并且react19的若干特性也会慢慢的在新架构实现。批处理已经实现了。

## 官方示例demo

[rn_test](https://github.com/facebook/react-native/blob/main/packages/rn-tester/js)

## 学习文档

[ReactNative](https://reactnative.dev/docs/getting-started)

## ReactNative库查询

[ReactNative library](https://reactnative.directory/)

## 问题解决路径

[ReactNative issues](https://github.com/facebook/react-native/issues/)

## 一套集成了expo的官方推荐开发模版（可以极大提高开发效率）

[ignite模版](https://github.com/infinitered/ignite)

## 三个react native最重要的库

[reactnavigation](https://reactnavigation.org/docs/getting-started)

[react-native-gesture-handler](https://docs.swmansion.com/react-native-gesture-handler/docs/)

[react-native-reanimated](https://docs.swmansion.com/react-native-reanimated/docs/fundamentals/getting-started)

## 一些常用的RN库

[代码示例](https://github.com/Hao-yiwen/reactNative-study): 代码示例，以下内容都可以在此找到示例。

[lottie-react-native](https://github.com/lottie-react-native/lottie-react-native): lottie动画，[lottie文件下载](https://lottiefiles.com/featured)

[react-native-snap-carousel](https://github.com/meliorence/react-native-snap-carousel): 轮播图

[react-native-video](https://github.com/react-native-video/react-native-video): Video

[rematch](https://github.com/rematch/rematch): 状态管理

[sqlite](https://github.com/andpor/react-native-sqlite-storage): sqlite3

[svg](https://github.com/software-mansion/react-native-svg): 支持svg写法

[svg-transform](https://github.com/kristerkari/react-native-svg-transformer): svg转换，可以直接使用svg文件，需要若干配置

[预制图标](https://github.com/oblador/react-native-vector-icons): react-native-vector-icons

[简单存储](https://github.com/react-native-async-storage/async-storage):async-storage

[路由](https://github.com/react-navigation/react-navigation): react-navigation

[react-native-safe-area-context](https://github.com/th3rdwave/react-native-safe-area-context): rn中的安全距离

[录音API](https://github.com/react-native-audio-toolkit/react-native-audio-toolkit)

[react-native-linear-gradient](https://github.com/react-native-linear-gradient/react-native-linear-gradient) 渐变色

[react-native-modals](https://github.com/jacklam718/react-native-modals?tab=readme-ov-file) 弹窗

[Native Base组件库](https://docs.nativebase.io/radio) 组件库 有Radio组件

[react-native-toast-message](https://github.com/calintamas/react-native-toast-message) Toast组件

[react-native-device-info](https://github.com/react-native-device-info/react-native-device-info) 获取设备信息

## workFlow

目前 RN 脚手架统一使用 ReactNative, cli 包在 ReactNative 核心仓库中，但是是由社区维护。

[react-native](https://github.com/facebook/react-native)

[@react-native-community/cli](https://github.com/react-native-community/cli)

:::tip
RN中初始化项目时，`Installing CocoaPods dependencies`会耗费大量时间。
:::
