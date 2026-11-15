---
title: Expo 框架与工作流全解析
sidebar_label: Expo 全解析
date: 2024-08-31
last_update:
  date: 2024-08-31
---

# Expo 框架与工作流全解析

从 React Native 0.74 开始，官方正式推荐 **Expo** 作为构建 RN 应用的首选框架。Expo 提供了完善的跨端 SDK、托管与原生工作流以及 EAS 云端构建服务，极大降低了移动端环境配置与构建成本。

- 官方文档：[Expo Documentation](https://docs.expo.dev/)
- 版本兼容对照：[Expo SDK Versions](https://docs.expo.dev/versions/latest/)

## 一、核心架构与工作流

### 1. 托管工作流（Managed Workflow / CNG）

- **快速启动**：无需在本地配置繁琐的 Xcode / Android Studio 原生环境即可快速初始化并预览应用。
- **开箱即用**：内置相机、定位、推送通知、音视频处理等常用原生能力，直接通过 JS API 调用。
- **持续原生生成（CNG / Prebuild）**：通过 `app.json` 与 Config Plugins 按需生成原生工程。

### 2. 裸工作流（Bare Workflow）

- **完全掌控原生层**：直接维护 `ios/` 与 `android/` 原生目录，可自由编写自定义 Native Modules，同时兼容绝大多数 Expo SDK 模块。

### 3. Expo Application Services (EAS)

- **EAS Build**：云端构建 iOS（`.ipa`）和 Android（`.apk` / `.aab`）安装包，配合 Expo Orbit 可直接拉取安装调试。
- **EAS Submit**：一键将构建产物提交至 App Store Connect 与 Google Play Console。
- **EAS Update**：遵循商店合规策略的 OTA 热更新服务，快速修复 JS 与静态资源问题。

---

## 二、存量裸 React Native 项目接入 Expo Modules

已有裸 RN 项目也可以按需引入 `expo-camera`、`expo-av` 等高质量模块（参考 [Installing Expo modules](https://docs.expo.dev/bare/installing-expo-modules/)）：

### 1. 自动安装基础桥接层

```bash
npx install-expo-modules@latest
```

### 2. 验证基础模块可用性

```js
import Constants from 'expo-constants';
console.log(Constants.systemFonts);
```

### 3. 按需安装具体功能包（以 `expo-camera` 为例）

```bash
npx expo install expo-camera
```

```tsx title="App.tsx"
import { Camera, CameraType } from 'expo-camera';
import { useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function App() {
    const [type, setType] = useState(CameraType.back);
    const [permission, requestPermission] = Camera.useCameraPermissions();

    if (!permission) {
        return <View />;
    }

    if (!permission.granted) {
        return (
            <View style={styles.container}>
                <Text style={{ textAlign: 'center' }}>需要相机权限才能预览画面</Text>
                <Button onPress={requestPermission} title="授予权限" />
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <Camera style={styles.camera} type={type}>
                <View style={styles.buttonContainer}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={() =>
                            setType(current =>
                                current === CameraType.back ? CameraType.front : CameraType.back
                            )
                        }
                    >
                        <Text style={styles.text}>切换镜头</Text>
                    </TouchableOpacity>
                </View>
            </Camera>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, justifyContent: 'center' },
    camera: { flex: 1 },
    buttonContainer: {
        flex: 1,
        flexDirection: 'row',
        backgroundColor: 'transparent',
        margin: 64,
    },
    button: { flex: 1, alignSelf: 'flex-end', alignItems: 'center' },
    text: { fontSize: 24, fontWeight: 'bold', color: 'white' },
});
```
