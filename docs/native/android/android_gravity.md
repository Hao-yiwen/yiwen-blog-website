---
title: gravity属性解释
sidebar_label: gravity属性解释
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# gravity属性解释

## 示例代码

```xml
<TextView
    android:layout_width="0dp"
    android:layout_height="wrap_content"
    android:layout_weight="1"
    android:gravity="center|right"
    android:text="总金额: "
    android:textColor="@color/black"
    android:textSize="17sp" />
```

## 解释
在 Android 的布局中，android:gravity 属性用于指定子视图在其容器内的对齐方式。这个属性可以应用于不同的容器，比如 LinearLayout、FrameLayout 或者任何其他的视图组件。android:gravity 的值可以是单个选项，也可以是多个选项的组合，这些选项通过竖线 (|) 分隔。

`android:gravity="center|right"`

这个特定的属性值 "center|right" 表示：

1. center：这部分意味着子视图在其容器的垂直方向上居中对齐。
2. right：这部分表示子视图在其容器的水平方向上靠右对齐。

当这两个选项组合在一起时，它们的意思是子视图在垂直方向上居中，同时在水平方向上靠右。换句话说，子视图将位于其父容器右侧的中间位置。