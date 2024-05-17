import android_cycle from '@site/static/img/20240304_android_cycle.png'

# Andoroid生命周期

Android 应用的生命周期是指应用在其运行期间经历的一系列状态变化，这些状态由操作系统通过调用 Activity 生命周期回调方法来管理。理解这些生命周期回调方法对于开发一个表现良好和用户友好的 Android 应用至关重要。以下是 Android Activity 生命周期的关键组成部分，以及每个阶段的主要职责：

<img src={android_cycle} width={400} />

## onCreate()：

应用创建时被调用。
用于进行一次性的初始化操作，比如设置用户界面布局（通过 setContentView()），数据绑定，初始化类级别的资源（如线程、数据库连接等）。

## onStart()：

当 Activity 对用户可见时，此方法被调用。
应用可以在此阶段恢复被 onPause() 暂停的资源更新。

## onResume()：

当 Activity 准备好与用户交互时，此方法被调用，此时 Activity 位于栈顶，并捕获所有用户输入。
应用可以在此阶段初始化那些只有在应用可见时才需要的资源。

## onPause()：

当系统准备启动或恢复另一个 Activity 时调用。
应用应该在这个方法中暂停影响应用可见性的任何更新，释放不需要的资源。

## onStop()：

当 Activity 不再对用户可见时，此方法被调用。
应用可以在此阶段释放或调整资源，数据同步操作通常在这里执行。

## onRestart()：

当当前 Activity 从停止状态重新启动进入运行状态时，此方法被调用。
应用可以在此阶段重新初始化在 onStop() 中释放的资源。

:::tip
onRestart() 方法上的星号表示，每次状态在 Created 和 Started 之间转换时，系统都不会调用此方法。仅当调用 onStop() 并且随后重启 activity 时，系统才会调用此方法。
:::

## onDestroy()：

当 Activity 即将被销毁时调用。
这是清理资源（如解绑服务或移除广播接收器）的地方。如果 Activity 被系统销毁（如旋转屏幕），可能不会总是被调用。

## 示例

### 进入app

-   onCreate() called
-   onStart() called
-   onResume() called

### app进入后台

-   onPause() called
-   onStop() called

### app从后台重新进入

-   onRestart() called(onRestart会在不触发oncreate的时候触发)
-   onStart() called
-   onResume() called

### app直接从后台杀死

-   会触发onPause,onStop，但是不会触发onDestory，因为被用户强制杀死进程，来不及触发ondestory

### 屏幕方向改变

-   onPause() called
-   onStop() called
-   onDestroy() called (会摧毁当前activity然后再oncreate)
-   onCreate() called
-   onStart() called
-   onResume() called

### 主题发生改变

-   onPause() called
-   onStop() called
-   onDestroy() called
-   onCreate() called
-   onStart() called
-   onResume() called

## 触发onCreated的场景

1. 更改应用主题颜色
2. 屏幕方向改变
