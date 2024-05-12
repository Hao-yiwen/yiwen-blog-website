# Android集成flutter作为一个页面

经过前端时间android集成rn时候的复杂操作，今天心血来潮，想要集成flutter到android，于是看官方文档进行集成，没想到半个小时就搞完了。(因为真的没什么好搞的~~),flutter确实很方便。

## 文档

[官方文档](https://docs.flutter.dev/add-to-app)

[博客](https://medium.com/@gabrielefeudo/integrate-flutter-app-in-native-android-app-a0b1ec105c38)

## 操作

本次集成使用aar集成，也就是随便在一个模块中开发好flutter，然后引入aar集成就可以。方便、简单~~

1. 创建flutter模块

```
flutter create -t module --org com.example my_flutter
```

2. 进行模块打包出aar文件

```
flutter build aar
```

3. 修改android项目的gradle

```gradle title="settings.gradle"
val storageUrl = System.getenv("FLUTTER_STORAGE_BASE_URL") ?: "https://storage.googleapis.com"

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)
    repositories {
        //... 省略
        maven {
            url =
                uri("/Users/haoyiwen/Documents/android/android-study/my_flutter/build/host/outputs/repo")
        }
        maven {
            url = uri("$storageUrl/download.flutter.io")
        }

    }
}
```

```gradle title="build.gradle"

dependencies {
    debugImplementation("com.example.my_flutter:flutter_debug:1.0")
    releaseImplementation("com.example.my_flutter:flutter_release:1.0")
}
```

4. 注册activity和从某一个activity跳转

```xml
<activity
  android:name="io.flutter.embedding.android.FlutterActivity"
  android:theme="@style/LaunchTheme"
  android:configChanges="orientation|keyboardHidden|keyboard|screenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
  android:hardwareAccelerated="true"
  android:windowSoftInputMode="adjustResize"
/>
```

```java
import io.flutter.embedding.android.FlutterActivity;

myButton.setOnClickListener {
  startActivity(
    FlutterActivity.createDefaultIntent(this)
  )
}
```
