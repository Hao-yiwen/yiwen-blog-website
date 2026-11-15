---
title: React Native 新旧架构演进读后笔记
authors: [yiwen]
tags: [React Native, 移动端架构, JSI]
date: 2024-04-12
---

import rn_old from '@site/static/img/rn_old.webp'
import rn_new from '@site/static/img/rn_new.webp'

晚上在 Medium 读到一篇剖析 React Native 新旧架构演进的长文，把耳熟能详却少有人讲透的底层痛点梳理得非常通透，特此整理核心脉络。

<!-- truncate -->

## 原文链接

- [React Native: The New Architecture (Medium)](https://medium.com/@under_the_hook/react-native-the-new-architecture-c4ba8ed8b452)

## 旧架构（Bridge 架构）核心瓶颈

<img src={rn_old} width={800} />

- **异步序列化通信**：强依赖 Bridge 通过 JSON 字符串异步消息队列传递数据，序列化/反序列化开销大。
- **双端隔离与内存冗余**：由于异步隔离，JS 侧和 Native 侧彼此不可见，不得不各自维护视图状态树并异步同步，带来额外内存开销。
- **单线程拥堵**：Bridge 通道负载过高时容易阻塞，导致交互卡顿。
- **长列表滑动白屏**：快速滚动列表时，JS 异步计算与跨桥渲染无法在同一帧内同步完成，容易出现短暂白屏。
- **启动全量加载**：App 启动阶段需要预先初始化全部 Native Modules，拖慢首屏启动时间。

## 新架构（JSI + Fabric + TurboModules）

<img src={rn_new} width={800} />

- **JSI（JavaScript Interface）**：基于 C++ 实现轻量通用互操作层，让 JS 可以持有 C++ HostObject 引用并直接同步或异步调用 Native 方法，彻底告别 JSON Bridge。
- **同步与实时互操作**：支持高优先级交互（如手势、滚动布局）同步渲染，消除异步延迟导致的白屏闪烁。
- **共享 C++ 阴影树（Fabric）**：C++ 层统一维护 Shadow Tree，大幅减少跨层复制与内存损耗。
- **按需懒加载（TurboModules）**：Native 模块只在真正被 JS 调用时才按需初始化，显著缩短冷启动耗时。
- **Codegen 静态类型安全**：通过 TypeScript / Flow 规范在构建期自动生成跨语言接口胶水代码，提升类型安全并降低运行时崩溃率。
