---
title: 动态设置RN端口和host
sidebar_label: 动态设置RN端口和host
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 动态设置RN端口和host

这是24年6月份的第一个提交，从五月底开始一直在解决RN动态切换开发服务器url问题，经过昨晚一整夜的折腾，以及今天一天的思考和发现，终于在6.2日解决了这个问题。

这个问题目前在社区并没有答案，我在想难道他们没有这样的场景吗，动态的设置RN端口和host。。。真是令人费解。

## 代码

```
PackagerConnectionSettings packagerConnectionSettings = new PackagerConnectionSettings(this);
packagerConnectionSettings.setDebugServerHost(debugServerHost);
```

在这里只要设置url和host就好。不知道为什么官方要将这个方法藏的如此的隐蔽，以至于需要找好长时间，默认是用8081端口，然后可以调用此方法修改。
