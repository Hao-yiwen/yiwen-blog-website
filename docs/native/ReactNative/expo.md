# Expo

谈到`RN`开发，`Expo`绝对是绕不过去的一环，凭借这一体化的服务和丰富的`SDK`，`Expo`在`ReactNative`社区有着巨大的影响力，以下是`Expo`的简单介绍和使用。

## Expo介绍

Expo SDK 是一组工具和服务的集合，旨在帮助开发者更快速、更轻松地构建 React Native 应用。它提供了一系列的 API 和组件，让开发者能够访问设备的原生功能，同时避免直接编写原生代码。以下是 Expo SDK 的主要特点和组成部分：(文档)[https://docs.expo.dev/]

### 主要特点
1. 快速启动：Expo 提供了一个快速启动新项目的工具，无需配置复杂的开发环境。
2. 跨平台：编写一次代码，可在 iOS 和 Android 平台上运行。
3. 原生 API 访问：通过 JavaScript 访问原生设备功能，如摄像头、地理位置、加速计等。
4. 无需原生编译：大多数情况下，无需编译原生代码，可以直接在模拟器或真机上预览和测试应用。
5. Over-the-Air Updates (OTA)：提供了推送代码更新到已发布应用的能力，无需通过应用商店的审核过程。
6. Expo Go 应用：提供了 Expo Go 应用，用于快速预览和测试 Expo 项目。

### 组件和服务
- UI 组件：Expo 提供了一系列预制的 UI 组件，如图标、地图、滑动器等。
- API：Expo SDK 包含了丰富的 API，用于访问设备功能和原生平台能力。
- CLI 工具：Expo CLI 是一个命令行工具，用于创建、运行和部署 Expo 项目。
- 开发者服务：Expo 提供了构建服务、更新服务等，帮助开发者更轻松地管理和发布他们的应用。
- Expo Application Services (EAS)：一系列高级服务，包括构建服务、提交服务等，提供了更多的控制和灵活性。

## RN项目引入Expo

裸RN项目可以使用部分`Expo`包和库，例如(expo-av),但是在引入前需要引入`Expo`库，以下是引入步骤。[RN安装Expo modules](https://docs.expo.dev/bare/installing-expo-modules/)

1. 自动安装
```bash
npx install-expo-modules@latest
```

2. 验证
```js
import Constants from 'expo-constants';
console.log(Constants.systemFonts);
```

3. 接着就可以引入例如`expo-camera`

```bash
npx expo install expo-camera
```

4. 运行示例代码
```ts
import { Camera, CameraType } from 'expo-camera';
import { useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();

  if (!permission) {
    // Camera permissions are still loading
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  function toggleCameraType() {
    setType(current => (current === CameraType.back ? CameraType.front : CameraType.back));
  }

  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraType}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});
```