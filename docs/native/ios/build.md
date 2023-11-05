---
sidebar_position: 2
---

# IOS打包

## IOS打出包的类型

### 模拟器

-   当你在 Xcode 中选择模拟器作为目标设备并构建应用时，Xcode 会编译出一个 x86_64（或 arm64，取决于你的Mac处理器类型）架构的应用程序包，这样的包是为了在 Mac 上模拟的 iOS 环境中运行优化的。

-   包后缀为 .app

### 真实IOS设备上的包

-   当你选择真实的 iPhone 或 iPad 作为目标设备并构建应用时，Xcode 会编译出 ARM 架构的应用程序包。无论是测试版还是发布版，最终都会被封装成 .ipa 格式的文件。

## 模拟器的包在哪里找

在Xcode中，当你构建并运行一个项目到模拟器上时，编译生成的应用程序包会存放在Derived Data目录中。如果你需要手动找到这个包，可以按照以下步骤操作：

1. 找到Derived Data目录:

-   打开Xcode。
-   在顶部菜单中选择“Xcode” > Settings...”。
-   点击“Locations”标签。
-   你会看到“Derived Data”旁边有一个路径，可以点击箭头图标来在Finder中打开这个目录。

2. 导航到具体的构建产物目录:

-   在Derived Data目录中，找到与你的项目名称相关联的文件夹。
-   打开该文件夹，并导航到Build/Products目录。
-   你会看到不同配置（如Debug或Release）和平台（如iphoneos或iphonesimulator）的构建输出。

3. 找到模拟器的应用程序包:

-   在iphonesimulator的目录下，你应该能找到以.app为后缀的文件夹。这就是模拟器版本的应用程序包。

4. 在模拟器上手动安装应用:
   如果需要，你可以通过xcrun命令行工具将.app包安装到模拟器上。打开终端，然后使用以下命令：

```bash
# <path-to-app>是你的.app文件夹的路径。如果模拟器已经启动，booted参数将安装应用到当前正在运行的模拟器。
xcrun simctl install booted <path-to-app>
```
