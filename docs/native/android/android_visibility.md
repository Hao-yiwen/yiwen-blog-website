# visibility解释

## 代码示例
```xml
<LinearLayout
    android:id="@+id/ll_content"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="vertical"
    android:visibility="visible"/>

 <LinearLayout
    android:id="@+id/ll_empty"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="vertical"
    android:visibility="gone"
    tools:visibility="visible">
```

## 解释说明
在 Android 布局文件中，android:visibility 和 tools:visibility 属性用于控制视图的可见性，但它们各自有不同的用途和作用范围：

### android:visibility

这是一个常用的属性，用于控制视图在运行时的可见性。它可以设置为以下值之一：

- visible：视图在界面上可见。
- invisible：视图在界面上不可见，但仍然占据布局空间。
- gone：视图在界面上不可见，且不占用布局空间。
这个属性直接影响你的应用在设备上运行时的用户界面。

### tools:visibility
tools:namespace 属性是 Android Studio 提供的一组工具属性，用于在设计时改善布局文件的编辑体验，但不会影响应用的实际运行行为。tools:visibility 特别用于在 Android Studio 的布局编辑器中预览视图的可见性状态，但不会影响实际运行的应用程序。

- 使用 tools:visibility 可以方便地在开发过程中预览布局中的不同状态，而无需通过改变代码来查看这些状态。

- tools:namespace 属性在编译 APK 时会被忽略，因此不会影响你的应用的实际行为。

在你的示例中：

- android:visibility="gone" 设置 ll_empty 在实际运行的应用中默认是不可见的。
- tools:visibility="visible" 仅用于在 Android Studio 的布局编辑器中预览 ll_empty 时使它可见，方便设计和调试。

这样做的好处是你可以在开发过程中查看那些在实际应用中默认不可见的视图元素，而不必更改它们在运行时的实际状态。