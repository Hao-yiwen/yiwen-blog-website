---
title: AppDelegate
sidebar_label: AppDelegate
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# AppDelegate

在UIKit中，AppDelegate是应用生命周期管理的重要部分。以下是一些常用的AppDelegate生命周期函数：

1. application(\_:didFinishLaunchingWithOptions:)

-   在应用启动时调用，用于初始化应用程序。例如设置初始视图控制器、配置全局设置等。

```swift
func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
    // 初始化代码
    return true
}
```

2. applicationDidBecomeActive(\_:)

-   当应用从非活动状态进入活动状态时调用。例如从后台或在用户锁屏解锁后恢复时。

```swift
func applicationDidBecomeActive(_ application: UIApplication) {
    // 应用变为活动状态
}
```

3. applicationWillResignActive(\_:)

-   当应用即将进入非活动状态时调用，例如来电或短信通知。

```swift
func applicationWillResignActive(_ application: UIApplication) {
    // 应用将进入非活动状态
}
```

4. applicationDidEnterBackground(\_:)

-   当应用进入后台时调用。用于释放资源、保存用户数据、无效计时器等。

```swift
func applicationDidEnterBackground(_ application: UIApplication) {
    // 应用进入后台
}
```

5. applicationWillEnterForeground(\_:)

-   当应用从后台进入前台时调用，通常用于撤销在applicationDidEnterBackground中做的更改。

```swift
func applicationWillEnterForeground(_ application: UIApplication) {
    // 应用将进入前台
}
```

6. applicationWillTerminate(\_:)

-   当应用即将终止时调用。用于保存数据和进行清理工作。

```swift
func applicationWillTerminate(_ application: UIApplication) {
    // 应用将终止
}
```

7. application(\_:configurationForConnecting:options:)

-   在iOS 13及更高版本中，用于处理场景的创建。返回一个UISceneConfiguration对象，用于配置新的场景。

```swift
func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
    return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
}
```
