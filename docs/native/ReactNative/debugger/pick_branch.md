---
title: 拉取需要的分支
sidebar_label: 拉取需要的分支
date: 2024-07-04
last_update:
  date: 2024-07-04
---

# 拉取需要的分支

rn源码仓库分支巨多，如果全部拉取搞不好会需要几个G的存储空间。所以按需拉取就显得很重要了。

## 拉取对应版本的RN分支

```bash title="获取远程所有tag的hash值"
git ls-remote --tags origin
```

```bash title="找到特定标签的哈希值并检出"
git fetch origin abcd1234
git checkout -b my-branch-name abcd1234
```
