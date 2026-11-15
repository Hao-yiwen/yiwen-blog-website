---
title: 创建 React Native 第三方库
sidebar_label: 创建 RN 三方库
date: 2024-07-10
last_update:
  date: 2024-07-10
---

# 创建 React Native 第三方库

## 官方脚手架文档

- [create-react-native-library (Builder Bob)](https://callstack.github.io/react-native-builder-bob/create)

## 初始化工程

### 1. 创建独立发布的第三方库

```bash
npx create-react-native-library@latest awesome-library
```

### 2. 在现有项目中创建本地模块库

```bash
npx create-react-native-library@latest awesome-library --local
```

## 本地联调方案

参考：[How to test the library in an app locally](https://callstack.github.io/react-native-builder-bob/faq#how-to-test-the-library-in-an-app-locally)

- 使用 `npm pack` 打出 `.tgz` 包后在宿主项目中本地安装测试。
- 使用轻量私有 npm 仓库 [Verdaccio](https://verdaccio.org/) 在本机模拟完整发包与安装流程。
