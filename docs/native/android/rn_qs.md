---
title: Android集成rn常见报错
sidebar_label: Android集成rn常见报错
date: 2024-06-25
last_update:
  date: 2025-02-14
---

# Android集成rn常见报错

## command node path问题

```bash
Error:Execution failed for task ':app:recordFilesBeforeBundleCommandDebug'.
> A problem occurred starting process 'command 'node''
```

-   解决:

```
./gradlew --stop
```

**如果以上命令还是不生效则尝试如下方案**

- 从终端启动 Android Studio：在终端中运行 open -a "Android Studio"，这样启动的 Android Studio 会继承终端的环境变量。
