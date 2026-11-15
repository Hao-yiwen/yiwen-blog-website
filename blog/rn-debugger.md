---
title: React Native 0.73.6 调试体验修复记录
authors: [yiwen]
tags: [React Native, 调试]
date: 2024-04-02
---

经历了 `0.72.x` 和 `0.73.0` 版本的调试折磨后，在 2024 年 4 月 1 日升级到 [RN 0.73.6](https://github.com/facebook/react-native/tree/v0.73.6) 尝试新版调试工具与 Hermes Debugger 时，终于能够流畅运行了。

<!-- truncate -->

一个无法正常断点调试的 React Native 项目实在让人崩溃。好在官方终于对这个问题进行了定位和修复，后续开发体验改善了很多。而且这次更新后，重新刷新页面也不会再导致 Debugger 连接失效。

## 注意事项

- 如果完全退出 App 后重新进入，需要重新连接调试工具，因为此时的 Source Map 实例已经更新。推荐操作顺序：先在 App 中打开对应的 RN 页面，再打开调试工具，这样 Source Map 就能正常映射连接。
