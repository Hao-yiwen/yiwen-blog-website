---
title: RN在andorid中的调试小技巧
sidebar_label: RN在andorid中的调试小技巧
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# RN在andorid中的调试小技巧

## 问题1

当电脑启动一个metro服务，然后再真机跑的时候真机如果和电脑不在同一个网络下面，则无法启动metro。

## 解决方案

使用`abd`反向代理

```bash
adb reverse tcp:8081 tcp:8081
```