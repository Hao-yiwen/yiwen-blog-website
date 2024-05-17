# Android侧设置图标

## 1. 使用Android Studio 的 Image Asset 工具

Android Studio 提供了一个便捷的工具来帮助你创建和管理应用图标，称为 Image Asset Studio。

1. 在 Android Studio 中，右击 res 文件夹，选择 New > Image Asset。
2. 在打开的 Image Asset Studio 中，选择 Launcher Icons (Adaptive and Legacy) 作为 Icon Type。
3. 上传你的图标图片，并根据需要调整选项（例如，裁剪、背景色等）。
4. 点击 Next，然后点击 Finish 来生成图标资源。
   这个工具会自动生成多个密度版本的图标，并放置在正确的文件夹中。

## 2. 更新 AndroidManifest.xml

```xml
<application
    android:icon="@mipmap/ic_launcher"
    android:roundIcon="@mipmap/ic_launcher_round"
    ... >
```