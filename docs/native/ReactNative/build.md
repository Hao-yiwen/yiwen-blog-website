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