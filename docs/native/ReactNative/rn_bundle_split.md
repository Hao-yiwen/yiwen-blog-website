# rn拆包

在大型app中，rn项目拆包变得十分必要，这篇文档将详细介绍分包及分包demo实践。

## 文档

[文档](https://juejin.cn/post/7355439624100249635?utm_source=nav&utm_medium=web&utm_campaign=coze0409)

## 实践

-   [0.72.5拆包](https://github.com/Hao-yiwen/reactNative-study/tree/master/splitRn_0725)

-   先进行common包的打包，其中要生成commoninfo.json，供业务bundle打包时候忽略该文件
-   业务bundle打除了commonbundle提供的文件之外的文件
-   android侧先加载common包，接着才会加载业务包

## 问题

-   [0.73.6拆包](https://github.com/Hao-yiwen/reactNative-study/tree/master/splitRn_0736

按照文档逻辑进行拆包，前期metro打包正常，但是在0.73.6+版本中遇到如下报错

```
FATAL EXCEPTION: mqt_js (Ask Gemini)
Process: com.splitrn_0736, PID: 23330
java.lang.RuntimeException: com.facebook.react.devsupport.JSException: Cannot read property 'setGlobalHandler' of undefined
    at com.facebook.react.bridge.DefaultJSExceptionHandler.handleException(DefaultJSExceptionHandler.java:20)
    at com.facebook.react.devsupport.DisabledDevSupportManager.handleException(DisabledDevSupportManager.java:195)
    at com.facebook.react.bridge.CatalystInstanceImpl.onNativeException(CatalystInstanceImpl.java:614)
    at com.facebook.react.bridge.CatalystInstanceImpl.-$$Nest$monNativeException(Unknown Source:0)
    at com.facebook.react.bridge.CatalystInstanceImpl$NativeExceptionHandler.handleException(CatalystInstanceImpl.java:632)
    at com.facebook.react.bridge.queue.MessageQueueThreadHandler.dispatchMessage(MessageQueueThreadHandler.java:40)
    at android.os.Looper.loopOnce(Looper.java:222)
    at android.os.Looper.loop(Looper.java:314)
    at com.facebook.react.bridge.queue.MessageQueueThreadImpl$4.run(MessageQueueThreadImpl.java:234)
    at java.lang.Thread.run(Thread.java:1012)
Caused by: com.facebook.react.devsupport.JSException: Cannot read property 'setGlobalHandler' of undefined
    at com.facebook.jni.NativeRunnable.run(Native Method)
    at android.os.Handler.handleCallback(Handler.java:958)
    at android.os.Handler.dispatchMessage(Handler.java:99)
    at com.facebook.react.bridge.queue.MessageQueueThreadHandler.dispatchMessage(MessageQueueThreadHandler.java:29)
    at android.os.Looper.loopOnce(Looper.java:222) 
    at android.os.Looper.loop(Looper.java:314) 
    at com.facebook.react.bridge.queue.MessageQueueThreadImpl$4.run(MessageQueueThreadImpl.java:234) 
    at java.lang.Thread.run(Thread.java:1012) 
Caused by: com.facebook.jni.CppException: Cannot read property 'setGlobalHandler' of undefined

TypeError: Cannot read property 'setGlobalHandler' of undefined
    at anonymous (common.android.bundle:157:150)
    at h (common.android.bundle:2:1789)
    at d (common.android.bundle:2:1250)
    at i (common.android.bundle:2:501)
    at anonymous (common.android.bundle:143:85)
    at h (common.android.bundle:2:1789)
    at d (common.android.bundle:2:1250)
    at i (common.android.bundle:2:501)
    at global (common.android.bundle:484:4)
    ... 8 more
```

该问题已反馈给`react native`，请持续关注进展和解决方案。
