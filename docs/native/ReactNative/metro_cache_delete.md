---
title: 删除metro缓存
sidebar_label: 删除metro缓存
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 删除metro缓存

## 1. 使用命令行直接清除缓存

当你启动Metro服务时，可以通过添加特定的命令行参数来清除缓存。以下命令会启动Metro bundler并清除其缓存：

```bash
npx react-native start --reset-cache
```

## 2.手动删除缓存文件

Metro bundler 的缓存文件通常存放在系统的临时文件夹中。你可以手动找到这些文件并删除它们。这个位置可能根据操作系统的不同而不同：

```bash
rm -f /tmp/metro-* & rm -rf $TMPDIR/metro-*
```

## 3.重置整个RN环境

```bash
rm -rf node_modules/ && npm cache clean --force && yarn && watchman watch-del-all && rm -f /tmp/metro-* & rm -rf $TMPDIR/metro-* && npx react-native start --reset-cache
```
