---
title: 如何在传统的用Java View写的Android项目中添加Kotlin
sidebar_label: 如何在传统的用Java View写的Android项目中添加Kotlin
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 如何在传统的用Java View写的Android项目中添加Kotlin

## 在项目级别的build.gradle.kts中添加插件

```kt
id("org.jetbrains.kotlin.android") version "1.9.0" apply false
```

## 在应用级别build.gradle.kts中引用插件和设置kotlin参数

```kt
plugins {
    // ...省略其他
    id("org.jetbrains.kotlin.android")
}

android {
    kotlinOptions {
        jvmTarget = "1.8"
    }
}
```

## 如果想在JavaView中添加Compose，则添加如下配置

```kt
composeOptions {
    kotlinCompilerExtensionVersion = "1.5.1"
}
```