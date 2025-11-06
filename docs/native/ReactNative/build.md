---
title: RN打包命令
sidebar_label: RN打包命令
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# RN打包命令

## Android 打包 JS Bundle

```bash
npx react-native bundle --platform android --dev false --entry-file index.js --bundle-output android/app/src/main/assets/index.android.bundle --assets-dest android/app/src/main/res
```

### 解释：
- --platform android：指定打包平台为 Android。
- --dev false：设置为生产模式，确保打包的 bundle 被优化。
- --entry-file index.js：指定入口文件，index.js 是默认的入口文件。
- --bundle-output：指定 bundle 文件的输出路径。对于 Android，通常是 android/app/src/main/assets/index.android.bundle。
- --assets-dest：指定资源（如图片和字体）的输出路径。对于 Android，通常是 android/app/src/main/res 目录。

## iOS 打包 JS Bundle

```ios
npx react-native bundle --platform ios --dev false --entry-file index.js --bundle-output ios/main.jsbundle --assets-dest ios
```

### 解释：
- --platform ios：指定打包平台为 iOS。
- --dev false：设置为生产模式，确保打包的 bundle 被优化。
- --entry-file index.js：指定入口文件，index.js 是默认的入口文件。
- --bundle-output：指定 bundle 文件的输出路径。对于 iOS，可以是 ios/main.jsbundle。
- --assets-dest：指定资源（如图片和字体）的输出路径。对于 iOS，通常是 ios 目录。

## hermes介绍
Hermes 是一个 JavaScript 引擎，专为运行在移动设备上的 React Native 应用优化。它旨在提高应用的启动速度，降低内存使用，并优化整体性能。为达到这些目标，Hermes 使用了一种名为 Hermes bytecode (HBC) 的中间字节码格式。

### Hermes Bytecode (HBC)
Hermes bytecode (HBC) 是 Hermes 编译器处理 JavaScript 代码后生成的二进制格式。与直接使用 JavaScript 源代码相比，使用 HBC 有几个显著的优点：

1. 启动性能提升：
使用 HBC 可以加快应用的启动时间。由于字节码更紧凑，加载和解析速度比普通的 JavaScript 快得多。
2. 内存占用减少：
HBC 文件通常比等效的 JavaScript 文件小，这意味着它们占用更少的内存。此外，Hermes 设计时就考虑到了内存效率，进一步减轻了内存压力。
3. 预编译优势：
JavaScript 源代码在执行前需要被解析和编译，这是一个耗时的过程。使用 HBC，这一步骤可以在应用构建阶段完成，从而减少运行时的计算需求。
4. 性能优化：
Hermes 对字节码执行了优化，以适应常见的 JavaScript 使用模式和 React Native 的特点，提高执行效率。

### 如何打包成hbc

每个rn版本的`node_modules/react-native/sdks/hermesc/osx-bin/hermes`可以将`jsbundle`转化为hbc格式文件。

所以可以在打包成jsbundle后再转化为hbc bundle。
