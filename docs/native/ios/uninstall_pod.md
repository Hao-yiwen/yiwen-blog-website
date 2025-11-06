---
title: 卸载pod
sidebar_label: 卸载pod
date: 2024-07-02
last_update:
  date: 2024-07-02
---

# 卸载pod

1. 查找所有相关pod依赖
```bash
gem list --local | grep cocoapods
```

2. 删除列出的所有cocoapods相关依赖,例如

```bash
gem install cocoapods-search
```