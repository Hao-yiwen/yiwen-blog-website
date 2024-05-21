# EventBus 概述
EventBus 是一个用于 Android 和 Java 的事件发布/订阅库。它简化了应用程序中组件之间的通信，尤其是当它们之间没有直接引用时。EventBus 提供了一种松耦合的方式，让组件可以通过发布和订阅事件来进行通信。

## EventBus 的主要特性
- 松耦合通信：发布者和订阅者彼此之间没有直接引用，减少了组件间的依赖性。
- 线程模型支持：支持在不同线程（主线程、后台线程、POSTING线程等）中发布和处理事件。
- Sticky 事件：可以发布 Sticky 事件，这样即使订阅者在事件发布之后才注册，也能接收到事件。
- 事件优先级：可以设置事件订阅的优先级，优先级高的订阅者会先处理事件。

## 代码示例

1. 添加依赖
在项目的 build.gradle 文件中添加 EventBus 的依赖：

```gradle
dependencies {
    implementation 'org.greenrobot:eventbus:3.3.1'
}
```

2. [代码示例](https://github.com/Hao-yiwen/android-study/blob/master/xml-and-compose-view-samples/java_view_other/src/main/java/com/yiwen/java_view_other/EventBusActivity.java)