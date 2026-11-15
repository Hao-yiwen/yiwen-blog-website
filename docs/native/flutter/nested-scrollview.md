---
title: 嵌套滚动
sidebar_label: 嵌套滚动
date: 2024-08-25
last_update:
  date: 2024-08-25
---

# 嵌套滚动

在开发的时候不可避免的就是可能会碰到滚动嵌套的问题。

-   如果是常规的在一个scrollview中嵌套多个scrollview，那么rn也是可以实现的。
-   如果想做一个类似于flutter中的tabview中里面的scrollview和外面的scrollview联动的效果就需要用 https://github.com/PedroBern/react-native-collapsible-tab-view 库了。
-   如果想在 header 部分做一个 scrollview，然后下面有一个 tabview，从上向下滚动到 tabview 的时候 tab 吸顶固定，然后内部 scrollview 继续滚动（外层与内层联动，视觉上如同同一个滚动容器），在 Flutter 中可以使用 `NestedScrollView` 轻松实现。

具体示例看flutter application中的demo实现。
