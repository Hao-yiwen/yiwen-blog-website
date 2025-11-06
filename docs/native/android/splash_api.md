---
title: Splash Api
sidebar_label: Splash Api
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Splash Api

Android 12引入的新的启动画面API（SplashScreen API）主要旨在简化应用的启动画面实现，确保启动画面的显示更加一致，并减少应用启动时间。这个API确实支持一些动画功能，但它的设计主要是用来显示静态图片或非常简单的动画，比如矢量图形的动画。

## 动画支持

新的SplashScreen API允许使用带有动画的图标，但这些动画应该是简单的，基于矢量图形的动画，使用的是AnimatedVectorDrawable。这种动画通常用于简单的图形变换，如旋转、渐变、路径变化等。

如果你的需求是显示复杂的动画，比如一个完整的动画序列、GIF或者视频，那么你可能需要采用传统的方式自定义启动画面。

## 文档

[文档](https://developer.android.com/develop/ui/views/launch/splash-screen?hl=zh-cn)

## 总结

今晚想做一个splash screen，本来以为splash api很强大，但是经过实际测试，发现功能还是很羸弱，如果想要设置splash screen，还是走activity创建比较好。
