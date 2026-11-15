---
title: react和reactDom
sidebar_label: react和reactDom
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# react和reactDom

在 React 生态系统中，react 和 react-dom 是两个核心的库，它们的分工是明确且互补的：

## react
- 核心库：react 是 React 的核心库，它提供了创建 React 组件和管理组件生命周期的能力。这个库包含了定义组件的基础要素，如 JSX、组件（类和函数式）、状态和生命周期方法等。
- 不依赖于特定环境：react 库是独立于任何特定平台的。它的职责是实现 React 的核心算法，处理组件的声明、状态管理、组件之间的组合以及其他基本的功能。
- 可用于多个平台：react 库的设计不仅适用于 Web 应用，还可以被用于其他平台（如 React Native）来创建原生移动应用。


## react-dom
- 平台特定库：react-dom 是专门为 Web 平台提供的库。它处理在浏览器中渲染 UI，并管理 DOM 的交互。react-dom 将 React 组件转换为 DOM 元素，并负责更新 UI 来响应数据的变化。
- 事件处理：react-dom 还负责处理浏览器的事件系统，它提供了一种方式来兼容不同浏览器的事件行为，并将其统一为 React 的事件处理系统。
- 与浏览器的交互：此外，react-dom 还负责其他与浏览器交互的功能，比如提供了用于服务器端渲染的 API (react-dom/server)。

## 总结
- react 提供了创建和管理 React 组件的基础能力，是构建 React 应用的核心。
- react-dom 专注于将 React 组件渲染到 Web 浏览器的 DOM 中，处理所有与 DOM 相关的操作和浏览器事件。
- 这种分离使得 React 能够跨平台使用，同时让 react 核心库保持轻量级和专注于通用的组件逻辑，而 react-dom 负责处理 Web 平台特有的实现细节。