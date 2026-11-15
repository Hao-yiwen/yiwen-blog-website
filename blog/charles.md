---
title: Charles 移动端抓包配置指南
authors: [yiwen]
tags: [Charles, Android, 抓包, 调试]
date: 2024-03-19
---

在从零开始开发 Android 项目、尚未接入完善网络调试基建时，如何直观查看接口请求与响应情况？重新梳理了一遍使用 Charles 进行移动端抓包的完整配置流程。

<!-- truncate -->

## 安装 Charles

前往官网下载安装包：[Charles Download](https://www.charlesproxy.com/download/latest-release/)

## 设置代理

1. 点击 `Proxy -> Proxy Settings` 配置代理端口，并勾选 `Enable transparent HTTP proxying`。
2. 点击 `Proxy -> SSL Proxying Settings`，启用 SSL 代理并添加需要解析的域名（或 `*:*`）。

## 安装根证书到 Android 设备

1. 在 Charles 中点击 `Help -> SSL Proxying -> Save Charles Root Certificate...`，将根证书导出为 `.cer` 文件。
2. 将 `.cer` 证书转换为 `.pem` 格式（可选，视系统要求）：

```bash
openssl x509 -inform der -in android.cer -out android.pem
```

3. 将证书推送到手机存储：

```bash
adb push android.pem /sdcard/
```

4. 在手机系统设置中从存储设备安装该证书，并将 Wi-Fi 代理指向电脑局域网 IP 与 Charles 监听端口，即可开始抓包。
