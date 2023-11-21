---
sidebar_position: 5
---

import android_proxy from "@site/static/img/android_proxy.png";

# Activity 模式

## 常见四种模式

在Android中，Activity有四种启动模式，分别是：

1. `standard`（标准模式）：这是默认的启动模式。每次启动Activity时，系统都会创建Activity的新实例，不管这个Activity是否已经存在。
2. `singleTop`（栈顶复用模式）：如果Activity已经位于任务栈的栈顶，那么再次启动这个Activity时，系统不会创建新的实例，而是复用栈顶的实例。如果Activity不在栈顶，系统会创建新的实例。 对应的Flag是`Intent.FLAG_ACTIVITY_SINGLE_TOP`。
3. `singleTask`（栈内复用模式）：系统会在任务栈中查找是否存在Activity的实例，如果存在就复用这个实例（会调用实例的onNewIntent()方法），并且会清除这个实例之上的所有Activity。如果不存在，系统会创建新的实例。 对应的Flag是`Intent.FLAG_ACTIVITY_CLEAR_TOP`或`Intent.FLAG_ACTIVITY_NEW_TASK`。
4. `singleInstance`（单实例模式）：在一个新的任务栈中创建Activity实例，并且这个任务栈中只有这一个Activity。对应的Flag是`Intent.FLAG_ACTIVITY_NEW_TASK`和`Intent.FLAG_ACTIVITY_MULTIPLE_TASK。`

这四种启动模式可以通过在AndroidManifest.xml文件中的`<activity>`标签的android:launchMode属性来设置。例如

```xml
<activity android:name=".MyActivity" android:launchMode="singleTop" />
```

在这个例子中，MyActivity的启动模式被设置为singleTop

## App内实现登录效果常用模式

`Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_NEW_TASK`: 会启动一个新的Activity，并清除同一任务栈中的所有旧Activity。(与`singleTask`有些相似，但是`singleTask`模式会复用任务栈中已经存在的`Activity`实例，而`FLAG_ACTIVITY_CLEAR_TASK | FLAG_ACTIVITY_NEW_TASK`会创建新的`Activity`实例)

## 启动标志的取值

-   `Intent.FLAG_ACTIVITY_NEW_TASK`： 开辟一个新的任务栈
-   `Intent.FLAG_ACTIVITY_SINGLE_TOP`：当栈顶为待跳转的活动实例之时，则重用栈顶的实例
-   `Intent.FLAG_ACTIVITY_CLEAR_TOP`：当栈中存在待跳转的活动实例时，则重新创建一个新实例，并清除原实例上方的所有实例
-   `Intent.FLAG_ACTIVITY_NO_HISTORY`：栈中不保存新启动的活动实例
-   `Intent.FLAG_ACTIVITY_CLEAR_TASK`：跳转到新页面时，栈中的原有实例都被清空
