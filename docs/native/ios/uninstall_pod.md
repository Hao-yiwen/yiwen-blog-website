---
title: 彻底卸载 CocoaPods
sidebar_label: 卸载 CocoaPods
date: 2024-07-02
last_update:
  date: 2024-07-02
---

# 彻底卸载 CocoaPods

当本地 Ruby / Gem 环境升级或需要切换为 Homebrew / Bundler 管理 CocoaPods 时，可按以下步骤清理全局 Gem 安装的旧版本：

1. 查看本地已安装的所有 `cocoapods` 相关 Gem 包：

```bash
gem list --local | grep cocoapods
```

2. 逐个（或批量）卸载列出的相关组件：

```bash
# 单个卸载示例
sudo gem uninstall cocoapods cocoapods-core cocoapods-deintegrate cocoapods-downloader cocoapods-plugins cocoapods-search cocoapods-stats cocoapods-trunk cocoapods-try

# 或通过管道一键批量卸载所有匹配项
gem list --local | grep cocoapods | awk '{print $1}' | xargs sudo gem uninstall -aIx
```
