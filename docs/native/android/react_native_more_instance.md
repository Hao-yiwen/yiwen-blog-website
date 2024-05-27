# Android实现rn多实例

最近想实现一下RN多实例，但是在实践的过程中一直有下面的报错,也就是在reactnative instance创建过程中一直有报错。

```
java.lang.NoClassDefFoundError: com.facebook.react.jscexecutor.JSCExecutor
```

刚开始认为是自己的多实例有问题，然后又研究了一些代码，但是回头看貌似没啥问题，然后就再github翻看了一下此类问题，果然找到答案了。以此来记录一下，否则后续再实践的过程中还是会经过遇到此问题。

-   [issues链接](https://github.com/facebook/react-native/issues/36048)
-   解决方案：

```java
ReactInstanceManager.builder()
// ...省略
.setJavaScriptExecutorFactory(HermesExecutorFactory())
```

也就是要在此处设置执行引擎。
