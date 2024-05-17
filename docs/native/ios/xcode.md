# Xcode常见命令

## xcode-select

xcode-select 用于管理机器上安装的Xcode版本。如果你有多个版本的Xcode安装，你可以使用 xcode-select 切换正在使用的版本。

```bash
# 查看当前使用的Xcode版本
xcode-select -p
# 切换Xcode版本
xcode-select -s /path/to/Xcode.app
```

## xcrun

xcrun 是一个运行Xcode和开发工具的命令行工具。你可以使用它来找到并运行Xcode中的工具，也可以用它来设置一些开发相关的配置。

```bash
# 查看应用大小报告：
xcrun xcsize --report <archive path>
```

### simctl

simctl 是 xcrun 的一部分，用于管理和控制iOS模拟器。你可以用它来安装和启动应用，获取设备列表，以及执行其他与模拟器相关的任务。

```bash
# 列出所有可用的设备:
xcrun simctl list devices
# 安装应用到模拟器:
xcrun simctl install <device_id> <app_path>
# 启动应用:
xcrun simctl launch <device_id> <bundle_id>
```

## xcodebuild

xcodebuild 是用于构建和编译Xcode项目和工作空间的命令行工具。你可以使用它来自动化构建过程，执行单元测试，创建应用的归档版本等。

```bash
# 构建项目或工作空间：
xcodebuild -project <projectname.xcodeproj> -scheme <schemeName> -configuration <Debug/Release>
# 运行测试：
xcodebuild test -workspace <workspaceName.xcworkspace> -scheme <schemeName>
# 创建归档版本：
xcodebuild -workspace <workspaceName.xcworkspace> -scheme <schemeName> -configuration <Debug/Release> archive -archivePath <archivePath>
# 导出IPA文件：
xcodebuild -exportArchive -archivePath <archivePath.xcarchive> -exportPath <exportPath> -exportOptionsPlist <exportOptionsPlist>
```
