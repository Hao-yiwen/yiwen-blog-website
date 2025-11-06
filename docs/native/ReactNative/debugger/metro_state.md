---
title: RN源码运行流程
sidebar_label: RN源码运行流程
date: 2024-07-10
last_update:
  date: 2024-10-29
---

# RN源码运行流程

:::danger
`npx react-native start`在0.75开始官方已经不维护了，官方希望使用expo脚手架来进行项目开发。
:::

## npx react-native bundle运行流程

-   @react-native-communtity/cli:社区来维护的处理 -> 运行RN项目的入口。

-   @react-native/cli-platform:官方维护的打包脚本 -> 此处有bundle的完整实现。
