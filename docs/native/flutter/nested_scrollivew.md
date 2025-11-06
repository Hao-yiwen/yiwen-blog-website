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
-   如果想在header部分做一个scrollview 然后下面有一个tabview，从上向下滚动到tabview的时候，tab固定，然后scrollivew继续滚动，实际上有外面的滚动和里面的两个滚动，但是看起来却像是一个滚动。目前rn还没有找到解决方案。

具体示例看flutter application中的demo实现。
