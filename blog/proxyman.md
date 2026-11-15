---
title: 使用 Proxyman 兼顾 iOS 抓包与外网代理
authors: [yiwen]
tags: [Proxyman, iOS, 抓包, 调试]
date: 2024-09-26
---

Web 开发可以直接用浏览器 DevTools 观察网络请求，而移动端开发在没有内置调试面板时，必须借助抓包工具。在 macOS 与 iOS 模拟器联调场景下，非常推荐使用 **Proxyman**。

<!-- truncate -->

## 为什么选择 Proxyman？

利用 Proxyman 的 **External Proxy（外部上游代理）** 功能，可以做到**一边正常访问外网查阅资料，一边同时拦截抓取模拟器与本机的请求**，彻底解决了“开抓包就断外网代理、开外网代理就抓不到包”的冲突。

## 外部代理配置要点

1. 打开 Proxyman 的 `Tools -> Proxy Settings -> External Proxy Settings`。
2. **不要勾选自动系统代理检测**（否则部分流量会被直接绕过）。
3. 手动勾选 `HTTP`、`HTTPS`（或 `SOCKS`），分别指向本机代理客户端监听的端口（如 `127.0.0.1:7890`）。

配置完成后，再也不用反复来回切换代理开关，可以流畅地一边调接口抓包、一边查文档了。
