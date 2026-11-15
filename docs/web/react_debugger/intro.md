---
title: React 18 源码调试准备与导读
sidebar_label: 源码调试准备
sidebar_position: 1
date: 2024-07-14
last_update:
  date: 2024-07-14
---

# React 18 源码调试准备与导读

React 作为现代前端里程碑式的框架，其 JSX 声明式抽象、跨平台 Reconciler 容器化解耦、批量更新、Fiber 时间切片与优先级调度思想深刻影响了 Web 与 React Native 体系。本专题围绕 `react@18.2.0` 客户端渲染主链路展开源码级追踪。

## 一、调试环境与源码仓库

- 调试搭建参考：[React 源码调试指南（掘金）](https://juejin.cn/post/7126501202866470949)
- 自维护带注释源码分支：[Hao-yiwen/react (`18.2.0-hyw-debug`)](https://github.com/Hao-yiwen/react/tree/18.2.0-hyw-debug)
- 调试宿主示例工程：[Hao-yiwen/react-debugger](https://github.com/Hao-yiwen/react-debugger)

## 二、VS Code 断点联调配置

在工程根目录配置 `.vscode/launch.json`，连接本地 Chrome 实例进行源码级单步断点追踪：

```json title=".vscode/launch.json"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch Chrome",
            "request": "launch",
            "type": "chrome",
            "url": "http://localhost:3000",
            "webRoot": "${workspaceFolder}/react-debugger"
        }
    ]
}
```

启动流程：

```bash
# 1. 在 react-debugger 目录启动开发服务器
yarn start

# 2. 在 VS Code 中按 F5 启动 Launch Chrome 调试会话
```
