---
title: 总结
sidebar_label: 总结
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 总结

现阶段大前端的实现中主要使用mvvm,因为大多数框架都做到了view和state的绑定。很少需要手动更新视图，除了古老的jqurey和手动操作dom需要。

## 常见的mvvm实现

- React Native + Redux 
- ios: uikit + viewmodel / compose + viewmodel
- android: view + viewmodel / swift + viewmodel

## view + viewmodel

[代码实践](https://github.com/Hao-yiwen/android-study/blob/master/xml-and-compose-view-samples/java_view_other/src/main/java/com/yiwen/java_view_other/LiveDataActivity.java)

数据源由livedata提供，需要用viewbinding绑定(如果view用纯代码实现，则用observe绑定)

## compose + viewModel

[代码实践](https://github.com/Hao-yiwen/GoalMan/tree/master/app/src/main/java/com/yiwen/goalman/ui/screen/Day)

其中数据源是flow提供的，以为flow是kt的特性之一。