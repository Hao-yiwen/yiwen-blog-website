---
title: RN缓存清理
sidebar_label: RN缓存清理
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# RN缓存清理

```bash
// 删除所有的watchman监听对象
// 删除所有metro的临时对象
watchman watch-del-all && rm -rf $TMPDIR/metro-* && rm -rf $TMPDIR/haste-map-*
```