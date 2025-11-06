---
title: watchman使用注意事项
sidebar_label: watchman使用注意事项
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# watchman使用注意事项

最近在使用watchman，但是电脑越变越卡，然后究其原因去查了一下，结果发现是watchman在监听一个文件夹后会一直监听，如果不手动清理则会一直运行，所以在电脑内存中经常看到watchman会占用很大内存。

## 解决办法

```bash
# 列出所有监听项
watchman watch-list

# 删除某个监听路径
watchman watch-del <path-to-your-project>

# 删除全部监听路径
watchman watch-del-all

# 重启 Watchman
watchman shutdown-server
```