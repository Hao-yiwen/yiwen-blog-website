---
title: 创建RN三方库
sidebar_label: 创建RN三方库
date: 2024-07-10
last_update:
  date: 2024-07-10
---

# 创建RN三方库

## 文档

[create-react-native-library](https://callstack.github.io/react-native-builder-bob/create)

## 初始化

### 创建线上库

``bash
npx create-react-native-library@latest awesome-library

````

### 创建本地库
```bash
npx create-react-native-library@latest awesome-library --local
````

## 本地调试

[本地调试](https://callstack.github.io/react-native-builder-bob/faq#how-to-test-the-library-in-an-app-locally)

-   使用`npm pack`是一种简单的方式
-   使用[Verdaccio](https://verdaccio.org/)也是一种较好的方案
