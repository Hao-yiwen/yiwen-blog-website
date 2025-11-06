---
title: PanResponder
sidebar_label: PanResponder
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# PanResponder

PanResponder 可将多次触摸调和为一个手势。它使单点触控手势不受额外触控的影响，并可用于识别基本的多点触控手势。

## FAQ

`PanResponder`在`ScrollView`中会出现手势透传问题，如何解决?

[Issues](https://github.com/facebook/react-native/issues/42008)

`Scrollview`中`scrollEnabled`为`false`时无法滚动，我们可以动态的在`PanResponder`开始滚动时设置为`true`,并在手势响应结束后设置为`false`,从而解决这个问题

```ts
PanResponder.create({
    onPanResponderStart: () => {setScrollEnabled(false)}
    onPanResponderMove: () => {
        // 更为精细化处理，在左右或上下滑动一段距离后再设置scrollEnabled
    },
    onPanResponderRelease: () => {setScrollEnabled(true)}
});
```
