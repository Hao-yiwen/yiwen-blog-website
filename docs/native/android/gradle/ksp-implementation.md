---
title: KSP
sidebar_label: KSP
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# KSP

KSP (Kotlin Symbol Processing) 是 Google 开发的一套工具，专门用于在编译时处理 Kotlin 代码的注解。与传统的注解处理器（如 kapt）相比，KSP 提供了更快的编译速度和更好的开发体验。使用 KSP，开发者可以编写自己的注解处理器来生成、修改或检查 Kotlin 代码。

- implementation 用于添加库依赖，使得库中的类和资源可以在项目的编译和运行时被使用。
- ksp 用于添加使用 KSP 处理注解的库，它专门处理 Kotlin 代码中的注解。使用 ksp 而不是 - implementation 添加这类依赖是因为，ksp 依赖仅用于编译时处理注解，而不会被包含在最终的 APK 或运行时环境中。

简而言之，如果你想要在 Kotlin 项目中使用注解处理器来生成代码或处理注解，就需要添加 ksp 的相关库来进行注解处理，这样做可以显著提高编译效率，并且只在编译时使用这些库，不会影响最终的运行时性能。

```kt
plugins {
    kotlin("jvm") version "1.4.32"
    id("com.google.devtools.ksp") version "<ksp-version>"
}

dependencies {
    implementation("...")
    ksp("...")
}
```

:::danger
KSP的使用要和kotlin版本对齐，否则会无法编译，并且有莫名其妙的报错原因。。。
:::