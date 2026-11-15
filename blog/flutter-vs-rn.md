---
title: 跨端选型思考：为什么个人 App 开发首选 Flutter 而非 RN
authors: [yiwen]
tags: [Flutter, React Native, 跨端开发, HarmonyOS]
date: 2024-07-28
---

在 HarmonyOS 成为国内第三极之前，移动端往往只需兼顾 iOS 与 Android 两端，咬咬牙还能控制在可预期的研发成本内；而当三端并存成为常态，在 React Native 与 Flutter 之间选定主力跨端生产力工具就变得势在必行。

<!-- truncate -->

## React Native 的工程代价

深入过 React Native 源码与 Android 原生集成后，对 RN 的实际开发体验颇有感触：

- **时刻伴随的性能顾虑**：面对复杂列表或高频手势场景，总要评估是否存在通信与渲染瓶颈，甚至需要手动编写原生组件或 Native Module 兜底。
- **重基建依赖**：RN 想要发挥上限，往往离不开完善的工程基建（如拆包方案、Bundle 预加载、热更新动态化等）；缺少基建支撑的裸 RN 开发体验存在不少掣肘。
- **多端一致性成本**：加上 HarmonyOS 适配层仍在演进，多端表现差异往往带来额外的兼容适配工作量。

## 为什么个人项目更适合 Flutter？

回归个人独立开发 App 的真实诉求——我其实并不强依赖 OTA 动态化下发，我真正需要的是**一套开箱即用、高性能、多端高度一致的 UI 生产力框架**：

1. **开发与调试体验流畅**：无需折腾复杂的拆包基建与历史包袱，专注于业务逻辑本身即可。
2. **原生级渲染性能**：基于自绘引擎与 Dart AOT 编译，常规业务场景下流畅度极高，甚至在复杂动效与拖拽交互（如内置 Drag & Drop 组件）上寥寥数行代码就能实现 RN 很难做顺滑的效果。
3. **FFI 原生互操作**：内置成熟的 `dart:ffi`，可以直接高效调用 C/C++ 高性能底层库。

参考性能评测对比：

- [Flutter vs Native vs React Native: Examining Performance](https://medium.com/swlh/flutter-vs-native-vs-react-native-examining-performance-31338f081980)
- [个人 Flutter 拖拽交互练习示例](https://github.com/Hao-yiwen/flutter_study/blob/master/flutter_application/lib/pages/drag/ExampleDragAndDrop.dart)
