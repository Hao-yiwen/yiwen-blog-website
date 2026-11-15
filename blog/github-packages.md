---
title: GitHub Container Registry (GHCR) 镜像推送鉴权踩坑
authors: [yiwen]
tags: [GitHub, Docker, GHCR]
date: 2023-09-29
---

在将 Docker 镜像推送至 GitHub Packages（`ghcr.io`）时遇到 `unauthorized: unauthenticated` 鉴权报错，记录原因与标准登录方式。

<!-- truncate -->

## 报错现象

执行 `docker push` 时返回未授权错误：

```text
Using default tag: latest
The push refers to repository [ghcr.io/owner/repo]
e07ee1baac5f: Preparing
unauthorized: unauthenticated: User cannot be authenticated with the token provided.
```

## 解决方案

参考 [GitHub Community 讨论 #50035](https://github.com/orgs/community/discussions/50035)：

1. 创建具备 `write:packages`、`read:packages`（及按需 `delete:packages`）权限的 Personal Access Token（Classic）。
2. 将 Token 存入环境变量并通过 `--password-stdin` 登录 `ghcr.io`：

```bash
export CR_PAT=YOUR_GITHUB_PAT
echo "$CR_PAT" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

登录成功后再执行 `docker push ghcr.io/owner/image:tag` 即可正常上传。
