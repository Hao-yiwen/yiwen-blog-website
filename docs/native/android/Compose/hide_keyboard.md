---
title: 键盘隐藏和焦点取消
sidebar_label: 键盘隐藏和焦点取消
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import keyboard_2024_0405 from "@site/static/img/2024_0405_keyboard.png";

# 键盘隐藏和焦点取消

## 场景

<img src={keyboard_2024_0405} width={300} />

从图片可以看到键盘和输入框之间是一个LazyColumn,所以我希望在我滚动的时候焦点和键盘能够自动消失，而不是还需要手动去控制，好在compose提供了简单方法处理这个问题。

## 解决办法

```kt
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController

val keyboardController = LocalSoftwareKeyboardController.current
val focusManager = LocalFocusManager.current

Column(modifier = Modifier
.pointerInput(Unit) {
    detectTapGestures(
        onPress = {
            focusManager.clearFocus()
            keyboardController?.hide()
        }
    )
}) {
    // ...lazycolumn布局
}
```

- 从中可以看到可以通过`LocalSoftwareKeyboardController.current`来获取键盘控制，然后`keyboardController?.hide()`隐藏键盘
- 从中可以看到可以通过`LocalFocusManager.current`来获取焦点，然后使用`focusManager.clearFocus()`来取消焦点
