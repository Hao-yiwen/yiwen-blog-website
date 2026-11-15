---
title: Apple Silicon Mac 安装 CentOS Stream 9 虚拟机
authors: [yiwen]
tags: [macOS, VMware Fusion, CentOS, Linux]
date: 2023-09-25
---

记录在 Apple Silicon（ARM 架构）Mac 上使用 VMware Fusion 免费个人版安装 CentOS Stream 9 虚拟机的流程。

<!-- truncate -->

:::info
2023 年 9 月 23 日才发现 macOS 上的 `VMware Fusion Player` 提供了个人免费授权版本（注：2024 年起 VMware Fusion Pro 也已对个人用户全面免费开放）。
:::

## 安装 VMware Fusion

前往官方页面下载安装 VMware Fusion（支持 Apple Silicon ARM64 架构）。

## 下载 CentOS Stream 9 镜像

前往[阿里云 CentOS Stream 9 (aarch64) 镜像源](https://mirrors.aliyun.com/centos-stream/9-stream/BaseOS/aarch64/iso/)，选择最新的 `dvd1.iso` 镜像下载并在 VMware Fusion 中新建虚拟机安装即可。

## 常用初始化配置

1. 终端体验优化：建议安装 `zsh` 与 `oh-my-zsh`，并配置 `zsh-autosuggestions` 和 `zsh-syntax-highlighting` 插件。
2. 基础开发工具：按需安装 `git` 与 `docker`。
