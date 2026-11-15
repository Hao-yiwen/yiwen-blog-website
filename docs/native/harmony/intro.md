---
title: HarmonyOS 架构概览与跨端生态
sidebar_label: HarmonyOS 介绍
sidebar_position: 1
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# HarmonyOS 架构概览与跨端生态

HarmonyOS（鸿蒙操作系统）是华为推出的面向全场景智能终端的分布式操作系统，覆盖智能手机、平板、穿戴设备、智慧屏与车载终端等多形态硬件。

## 核心架构特性

1. **全场景分布式能力**：通过分布式软总线、分布式数据管理与任务调度，实现跨设备协同流转。
2. **微内核与安全隔离**：核心内聚精简，将非核心系统服务置于用户态隔离运行，提升安全性与可扩展性。
3. **声明式原生开发（ArkTS / ArkUI）**：基于 TypeScript 扩展的静态类型语言 ArkTS 配合声明式 UI 框架 ArkUI，开发范式与 SwiftUI / Jetpack Compose 一脉相承。
4. **OpenHarmony 开源底座**：底层核心能力开源，支持广泛的行业终端与三方生态共建。

## React Native 鸿蒙适配（RNOH）生态

针对存量 React Native 应用迁移至 HarmonyOS NEXT，社区与官方维护了专门的鸿蒙三方库兼容组织：

- [react-native-oh-library（React Native OpenHarmony 适配三方库合集）](https://github.com/react-native-oh-library)
