---
title: React Native 0.74 Android Release 包体积排查记录
authors: [yiwen]
tags: [React Native, Android, 性能优化]
date: 2024-04-27
---

import Size0740 from '@site/static/img/0740_size.png'
import Size0725 from '@site/static/img/0725_size.png'

晚上尝试对一个刚初始化的 `react-native@0.74.0` 空项目打 Android Release 包时，发现生成的 APK 文件显示体积竟然高达 50MB，着实吓了一跳。

<!-- truncate -->

## 0.74.0 打包体积对比

<img src={Size0740} width={500} />

对比此前 `0.72.5` 版本的打包结果：

<img src={Size0725} width={500} />

## 问题追踪与官方解答（2024-04-28）

随后向官方仓库提交了 Issue：[facebook/react-native#44291](https://github.com/facebook/react-native/issues/44291#issuecomment-2079889795)。

React Native 团队成员很快给出了[详细解答](https://github.com/facebook/react-native/issues/44291#issuecomment-2084952099)：

- **原因**：从 Android Gradle Plugin 新规范开始，`extractNativeLibs=false` 默认将 `.so` 原生库以**未压缩对齐**方式直接打包进 APK，以便系统直接通过 `mmap` 从 APK 中读取而无需安装后二次解压，从而减少用户手机上的实际安装占用空间并加快启动速度。
- **实际分发体积**：本地看到的通用 APK 体积变大只是因为未压缩且包含了全部 CPU 架构（`arm64-v8a`、`armeabi-v7a`、`x86`、`x86_64`）；在通过 AAB（Android App Bundle）上架应用商店或开启 ABI 分包与传输压缩后，用户实际下载体积并未增大。
