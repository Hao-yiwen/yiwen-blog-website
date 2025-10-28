---
title: charles使用
date: 2024-03-19
---

# charles使用

今天在开发android，在开发过程中在思考一个问题，如果是从0开始，并没有完善的基建，那么如何知晓接口返回的情况那，于是打算重新熟悉charles进行抓包

## 安装charles

https://www.charlesproxy.com/download/latest-release/

## 设置代理

-   点击`proxy-> proxy setting`来创建代理端口，点击`enable transparent http proxying`
-   点击`proxy-> ssl proxy setting`来容许所有流量流入

## 下载根证书到android手机

-   查看根证书在本地的位置，并导出证书为cer文件

-   将cer文件转化为pem文件

```bash
openssl x509 -inform der -in android.cer -out android.pem
```

-   将pem文件传输到手机

```bash
adb push android.pem /sdcard/
```

-   安装证书

接着就可以开始抓包了。
