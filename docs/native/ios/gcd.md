# GCD

Grand Central Dispatch (GCD) 是 Apple 为并发编程提供的一套高效的解决方案。GCD 允许开发者在多个队列中执行任务，从而充分利用多核处理器的性能，避免编写复杂的线程管理代码。GCD是Apple为了简化多线程开发推出的基础库，不是swift和oc的语言能力。(库: Dispatch)

## GCD 的优点

1. 简化并发编程：通过队列管理任务，而非直接管理线程。
2. 性能优化：自动调整系统资源使用，充分利用多核处理器。
3. 提高响应速度：将耗时操作放在后台线程，保持主线程响应。

## GCD常用线程和队列

在 iOS 中，Grand Central Dispatch (GCD) 提供了多种优先级的全局队列 (Global Dispatch Queues)。这些队列根据优先级 (QoS, Quality of Service) 来区分，主要包括以下几种：

1. User-Interactive (用户交互)

-   最高优先级，用于需要立即执行并确保快速完成的任务，通常用于更新 UI。

2. User-Initiated (用户发起)

-   用于用户直接发起的任务，需要在短时间内完成，例如加载数据。

3. Utility (实用)

-   中等优先级，适用于需要执行较长时间的后台任务，例如下载文件、数据处理等。

4. Background (后台)

-   最低优先级，用于对时间要求不高的后台任务，例如定期数据同步、预加载等。

5. Default (默认)

-   优先级介于 User-Initiated 和 Utility 之间，用于普通的任务。

6. 主线程 (Main Thread)

-   处理所有的 UI 更新和用户交互。
-   是应用启动时创建的默认线程。

```swift
// User-Interactive
DispatchQueue.global(qos: .userInteractive).async {
    // 高优先级任务
}

// User-Initiated
DispatchQueue.global(qos: .userInitiated).async {
    // 用户发起的任务
}

// Utility
DispatchQueue.global(qos: .utility).async {
    // 实用任务
}

// Background
DispatchQueue.global(qos: .background).async {
    // 后台任务
}

// 主线程更新
DispatchQueue.main.async {
    // 更新 UI 的代码
}
```

## 常用GCD任务

1. 异步任务

异步任务允许任务在后台执行，不阻塞主线程。

```swift
DispatchQueue.global(qos: .background).async {
    // 执行一些后台任务
    DispatchQueue.main.async {
        // 回到主线程更新 UI
    }
}
```

2. 同步任务

同步任务会阻塞当前线程，直到任务完成。注意：不要在主线程上使用同步任务，否则会导致应用无响应。

```swift
DispatchQueue.global(qos: .default).sync {
    // 执行一些同步任务
}
```

3. 主线程任务

在主线程上执行任务，常用于更新 UI。

```swift
DispatchQueue.main.async {
    // 在主线程上更新 UI
}
```

4. 延迟任务

延迟任务允许在指定的时间之后执行任务。

```swift
DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
    // 延迟 2 秒后执行任务
}
```
