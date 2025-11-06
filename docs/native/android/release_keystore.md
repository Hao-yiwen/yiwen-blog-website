---
title: Android 应用的签名
sidebar_label: Android 应用的签名
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android 应用的签名

Android 应用的签名机制是一个关键的安全特性，它确保了应用的安全性、完整性和来源验证。下面我将详细介绍 Android 应用签名的作用以及如何进行签名。

## 签名的作用

1. 完整性验证：
   应用签名确保自发布以来应用未被篡改。在安装或更新应用时，Android 系统会验证 APK 文件的签名。如果文件被修改过，签名将不匹配，系统将拒绝安装。
2. 来源验证：
   签名证明应用来自声明的开发者。这是通过使用开发者的私钥进行签名来实现的。只有拥有对应私钥的开发者才能生成有效的签名，用户可以通过这一点来验证应用的真实来源。
3. 更新控制：
   Android 要求对同一应用的所有更新版本使用相同的签名。这防止了未经授权的开发者发布恶意更新。
4. 权限请求：
   对于使用特定权限或功能的应用，如系统级服务，签名还可以用来限制访问。例如，某些 Android API 只允许签名过的应用或者与系统签名相同的应用调用。

## 如何生成签名

### 使用 Android Studio

1. 生成密钥库（Keystore）：
   在 `Android Studio 中`，可以通过 `Build > Generate Signed Bundle / APK` 来创建一个新的密钥库或使用现有的密钥库。你需要提供密钥库密码、密钥别名、密钥密码等信息。

### 使用命令行（使用 apksigner 工具）

1. 生成密钥库：

使用 keytool，这是 Java 开发工具包中的一部分。打开命令行或终端，执行以下命令：

```bash
keytool -genkey -v -keystore my-release-key.keystore -alias my-alias -keyalg RSA -keysize 2048 -validity 10000
```

这将提示你输入密钥库和密钥的密码以及一些证书信息。

### gralde中release配置

```gradle
android {
    ...
    signingConfigs {
        release {
            storeFile file("my-release-key.keystore")
            storePassword "store-password"
            keyAlias "my-key-alias"
            keyPassword "key-password"
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### 验证 APK是否签名成功

验证签名是否成功。

```bash
apksigner verify my-app.apk
```
