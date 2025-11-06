---
title: 如何在android中使用svg图标
sidebar_label: 如何在android中使用svg图标
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 如何在android中使用svg图标

## 转换svg为Vector Drawable

- 选择`New > Vector Asset`进行转换

## 使用Image组件添加
```kotlin
Image(
    painter = painterResource(id = R.drawable.vector_drawable), // 替换为你的资源ID
    contentDescription = "描述你的图形", // 提供无障碍支持的文本描述
    modifier = Modifier.fillMaxSize() // 或其他Modifier来调整显示方式
)
```