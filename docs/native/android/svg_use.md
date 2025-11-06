---
title: Android使用svg
sidebar_label: Android使用svg
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android使用svg

在 `Android` 开发中，使用 SVG（可缩放矢量图形）文件可以提供更清晰、更灵活的图形展示。不过，Android 原生并不直接支持 `SVG` 文件格式。为了在 `Android` 应用中使用 `SVG`，你可以采取以下几种方法：

## 将 SVG 转换为 Vector Drawable

`Android` 支持一种名为 Vector Drawable 的格式，它与 `SVG` 非常相似。你可以将 `SVG` 文件转换为 `Vector Drawable` 格式，然后像使用其他 `Drawable` 资源一样在你的应用中使用它们。

- 手动转换：可以使用 `Android Studio` 自带的 `Vector Asset Studio` 来将 `SVG` 文件转换为 `Vector Drawable`。
1. 右键点击 `res` 目录 -> `New` -> `Vector Asset`。
2. 选择 `Local file (SVG, PSD)`，然后导入你的 `SVG` 文件。
3. 跟随向导完成转换。
- 在线工具：也有许多在线工具可以将 `SVG` 文件转换为 `Vector Drawable` 格式。

## 将 SVG 转换为 PNG

直接将svg拖动到`drawable`目录下，邮件点击`convert to png`即可转化为`png`格式图片。这是一种不那么理想的方法，因为它牺牲了 `SVG` 的矢量特性。但如果你需要快速解决方案，可以将 `SVG` 文件转换为 `PNG` 格式，并将它们作为普通的图片资源使用。