---
title: 常见adb命令
sidebar_label: 常见adb命令
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 常见adb命令

`Android Debug Bridge (ADB)` 是一个强大的工具，用于管理和调试 Android 设备。以下是一些常见的 ADB 命令：

1. 查看连接的设备

`adb devices`：列出所有连接的设备和模拟器。

2. 安装和卸载应用

`adb install <apk路径>`：在设备上安装 APK。
`adb install -r <apk路径>`：重新安装 APK，保留数据。
`adb uninstall <包名>`：从设备上卸载一个应用。

3. 复制文件

`adb push <本地路径> <设备路径>`：将文件从本地复制到设备。
`adb pull <设备路径> <本地路径>`：将文件从设备复制到本地。

4. 日志和调试

`adb logcat`：查看设备的日志信息。
`adb bugreport`：收集设备的全面报告。

5. 管理设备

`adb reboot`：重启连接的设备或模拟器。
`adb shell`：启动一个远程 shell 在设备上执行命令。
`adb shell input keyevent <键值>`：模拟按键事件。
`adb shell am start -n <包名>/<活动名>`：启动一个应用的特定活动。

6. 管理端口转发

`adb forward --list`：列出所有转发的端口。
`adb forward <本地端口> <设备端口>`：设置端口转发。

7. 截屏和录屏

`adb shell screencap -p /sdcard/screen.png`：截取屏幕。
`adb shell screenrecord /sdcard/video.mp4`：录制设备屏幕。

8. 发送广播

`adb shell am broadcast -a <广播动作>`：发送一个广播。

9. 清理应用数据

`adb shell pm clear <包名>`：清除应用的数据。

10. 获取设备信息

`adb get-serialno`：获取设备的序列号。
`adb get-state`：获取设备的状态。

## 注意

使用 ADB 命令时，请确保你的设备已经开启了 USB 调试模式，并且已经正确地连接到你的电脑。部分命令可能需要设备的 root 权限才能执行。在执行某些操作（如安装 APK、清理应用数据）时，请确保你了解这些操作的后果。