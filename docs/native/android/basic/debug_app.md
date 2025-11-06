---
title: 如何分离release app和debug app
sidebar_label: 如何分离release app和debug app
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 如何分离release app和debug app

在app的入口module的build.gradle中添加以下applicationIdSuffix,通过使得debug和release的applicationid不同来打包出不同的app

```
debug {
    applicationIdSuffix = ".debug"
}
```
