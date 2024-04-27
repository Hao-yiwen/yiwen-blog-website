# FrameLayout

FrameLayout 是 Android 开发中的一个基本布局容器，它被设计为一个轻量级的容器，用来存放单个子视图或者视图堆栈。在 FrameLayout 中，子视图按照它们被添加的顺序堆叠在一起，最新添加的子视图会显示在最上层。这使得 FrameLayout 非常适合用来覆盖视图。

## 主要特点：

-   简单性：用于放置单个子视图，使其填满屏幕。
-   堆叠：可以堆叠多个子视图，后添加的视图会覆盖前面的视图。
-   用途：常用于加载片段、显示加载指示器、作为其他复杂布局的容器等。

## 使用场景：

-   覆盖元素：如在图片上方显示一个半透明的状态栏。
-   片段切换：在同一位置交替显示不同的片段。
-   加载指示器：在内容加载时覆盖内容显示进度环。

## 示例代码
```xml
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:id="@+id/goalman_widget_container"
    android:layout_height="match_parent">

    <ImageView
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:scaleType="centerCrop"
        android:src="@drawable/flower_78" />
    
    <TextView 
        android:text="test"
        android:layout_width="wrap_parent"
        android:layout_height="wrap_parent"/>
```

上述代码是在图片上面防止一段问题。
