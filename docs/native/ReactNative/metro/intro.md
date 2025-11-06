---
title: Metro
sidebar_label: Metro
date: 2024-09-23
last_update:
  date: 2024-09-23
---

# Metro

本章用来介绍metro打包过程，包括resolve/transform，以及序列化过程。

## 背景

因为最近在做metro的tree shaking功能，而该功能一直是rn中缺失的功能，恰好在年中7月中旬，expo实现了一版tree shaking功能，但是expo因为tree shaking高度定制了expo打包逻辑，而我将其迁移到普通的metro打包流程是有一定困难的，所以在这里记录一些遇到的问题。
