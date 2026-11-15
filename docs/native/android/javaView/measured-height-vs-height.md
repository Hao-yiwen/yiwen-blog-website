---
title: measureHeight和height
sidebar_label: measureHeight和height
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# measureHeight和height

在Android开发中，measureHeight和height是与视图（View）尺寸相关的两个概念，它们代表了视图的高度信息，但是用途和含义有所不同。

## measureHeight
- measureHeight是视图在测量阶段（measure pass）确定的高度值。
- 它是通过调用View.measure(int widthMeasureSpec, int heightMeasureSpec)方法计算得出的，并且可以通过View.getMeasuredHeight()方法来获取。
- measureHeight的值是根据视图的布局参数（如match_parent、wrap_content或具体的dp值）、视图的onMeasure方法逻辑以及父视图的尺寸约束计算得出的。
- 这个值表示视图希望在布局中占用的高度，但最终的实际高度还需要考虑父视图的布局策略和其他子视图的尺寸。
## height
- height是视图在布局阶段（layout pass）后的实际高度。
- 这个值可以通过View.getHeight()方法来获取。
- height的值是在布局阶段确定的，此时父视图会根据所有子视图的measureHeight值、自身的布局策略（如权重、对齐方式等）和可用空间来最终确定每个子视图的位置和尺寸。
- height表示视图在屏幕上实际占用的高度，这个值可能与measureHeight不同，因为父视图在布局过程中可能会调整子视图的尺寸以满足特定的布局要求。

## 总结
总的来说，measureHeight是视图在测量阶段希望达到的高度，而height是视图在布局完成后实际的高度。在开发过程中，通常是在需要计算视图尺寸或动态调整视图大小时会用到这些值。