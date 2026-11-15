---
title: TouchableWithoutFeedback
sidebar_label: TouchableWithoutFeedback
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# TouchableWithoutFeedback

```tsx
<TouchableHighlight onPress={() => {}} >
  <Text>Home</Text>
</TouchableHighlight>
```

## 为什么这段代码中，在`TouchableHighlight`被按下时候有灰色椭圆突出显示文字？

这是`Text`中的`suppressHighlighting`属性导致的，只针对`ios`有效，默认是有灰色椭圆效果，可添加`suppressHighlighting`属性来修改`Text`按下后默认展示效果。

```tsx
<TouchableWithoutFeedback onPress={() => {}}>
    <Text suppressHighlighting>Home</Text>
</TouchableWithoutFeedback>
```