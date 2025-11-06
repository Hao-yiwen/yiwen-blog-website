---
title: 开始
sidebar_label: 开始
date: 2024-07-14
last_update:
  date: 2024-07-14
---

# 开始

工作三年了，一直想系统性梳理一下React源码。在整个前端体系，React始终占据重要地位，无论是jsx，容器化，还是批处理，fiber切片，优先级调度。

React框架作为前端里程碑式的框架，无论是对Web开发，还是对Native开发，都产生了极大的影响，而在其内部，一套脱离于平台，立足于js的设计思想，在整个前端体系都有巨大的影响。我将根据自己的理解，在这里浅谈对React框架的理解。

为什么要进行react分析那，在工作一年多之后，我一直在思考我到底是一个前端还是一个移动端开发者，因为在前两年，我始终接触的是react-native。会写一些前端，但是大都是高度框架化的，既没有three.js也没有复杂的动画。所以在当时决心成为一个移动端开发者，我系统学习了整个react-native架构，了解了整个react-native的体系。因为是移动端，完整学习了android，并且学习了部分的ios。

今天我想系统性的来梳理一下react-native架构，因为这是我出发和赖以吃饭的东西。但是在开始之前，我需要大致了解和调试一些react源码。因为react-native是基于此开发的。让我们出发吧～

声明：因为现在主要做移动端工作，所以了解react在web端的源码只是为了更好的理解react-native体系，所以调试针对react-natived的客服端渲染，部涉及水合之类api。。。(等后续有时间再了解吧~~~)

## 调试准备工作

[调试参考文档](https://juejin.cn/post/7126501202866470949)

[react源码调试分支](https://github.com/Hao-yiwen/react/tree/18.2.0-hyw-debug)

[reactdebugger分支](https://github.com/Hao-yiwen/react-debugger)