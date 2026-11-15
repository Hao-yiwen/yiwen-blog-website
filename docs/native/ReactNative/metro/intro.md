---
title: Metro 打包器架构与缓存管理
sidebar_label: 架构与缓存管理
sidebar_position: 1
date: 2024-07-04
last_update:
  date: 2024-09-23
---

# Metro 打包器架构与缓存管理

Metro 是 React Native 官方 JavaScript 打包器（Bundler），负责模块依赖解析、Babel 代码转译、Bundle 序列化输出，以及开发期的 Fast Refresh 热更新服务。

- [Metro 官方配置文档](https://metrobundler.dev/docs/configuration#resolver-options)
- [AST Explorer（编写自定义 Babel 插件必备）](https://astexplorer.net/)

## 一、核心阶段与架构组成

1. **Resolver（模块解析器）**：解析模块导入路径、平台后缀（如 `.ios.tsx` / `.android.tsx`）、路径别名与 `package.json` 入口字段。
2. **Transformer（代码转换器）**：基于 Babel（或 SWC）将 TypeScript / JSX / ESNext 语法转译为目标运行时可执行代码，支持自定义 AST 插件。
3. **Serializer（序列化器）**：将模块依赖图（Dependency Graph）串联打包为一个或多个 Bundle 文件，也是实现拆包（Split Bundling）与自定义 Tree Shaking 的关键切入点。
4. **Server（开发服务器）**：在本地提供增量编译、Source Map 映射与 Fast Refresh 通信。

:::note 关于 Tree Shaking
传统 Metro 默认基于 CommonJS 风格模块封装，长期缺乏开箱即用的 Tree Shaking 能力；Expo CLI 在 SDK 51+ 基于自定义图分析实现了实验性 Tree Shaking，但将其平迁到通用裸 Metro 流水线仍需对 `Serializer` 与模块副作用标记做深度适配。
:::

## 二、Metro 与 Watchman 缓存清理指南

当修改了 `babel.config.js`、`metro.config.js` 或遇到诡异的模块解析缓存残留时，可按需分级清理缓存：

### 1. 启动时重置 Metro 转换缓存

```bash
npx react-native start --reset-cache
```

### 2. 清理 Watchman 监听与系统临时目录缓存

```bash
watchman watch-del-all && rm -rf $TMPDIR/metro-* && rm -rf $TMPDIR/haste-map-*
```

### 3. 彻底重置 RN 依赖与缓存环境

```bash
rm -rf node_modules/ \
  && watchman watch-del-all \
  && rm -rf /tmp/metro-* $TMPDIR/metro-* $TMPDIR/haste-map-* \
  && yarn install \
  && npx react-native start --reset-cache
```
