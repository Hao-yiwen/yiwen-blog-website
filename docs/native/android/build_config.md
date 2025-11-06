---
title: Android项目中使用BuildConfig
sidebar_label: Android项目中使用BuildConfig
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android项目中使用BuildConfig

[参考文档](https://developermemos.com/posts/jetpack-compose-buildconfig-import-issue)

## 为什么设置buildConfig

在有些时候有些代码只在开发时候使用，所以通过buildConfig配置可以在打包的时候处理那些代码是debug，那些是release。可以做差异化代码，最简单的示例就是ReactNative调试工具只在debugger时候使用。

## 步骤

-   在项目级别的`build.gradle`设置`buildConfig = true`

```java
buildFeatures {
    // The first line should already be in your project!
    compose = true
    buildConfig = true
}
```

- 引用BuildConfig开始使用，使用示例如下
```kotlin
import com.example.goalman.BuildConfig

if (BuildConfig.DEBUG) {
    Text(
        text = stringResource(id = R.string.title_name) + "(DEBUG)",
        style = MaterialTheme.typography.displayLarge
    )
} else {
    Text(
        text = stringResource(id = R.string.title_name),
        style = MaterialTheme.typography.displayLarge
    )
}
```