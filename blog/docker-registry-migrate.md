---
title: Docker 镜像仓库迁移至阿里云 ACR
authors: [yiwen]
tags: [Docker, GitHub Actions, CI/CD, 阿里云]
date: 2024-01-14
---

记录通过 GitHub Actions 自动构建 Docker 镜像并推送至阿里云容器镜像服务（ACR）的配置流程，解决国内服务器拉取 Docker Hub 镜像缓慢或超时的问题。

<!-- truncate -->

## 前置条件

1. 开通[阿里云容器镜像服务（ACR）](https://cr.console.aliyun.com/cn-shanghai/instance/repositories)并创建命名空间与镜像仓库。
2. 在 GitHub 仓库的 `Settings -> Secrets and variables -> Actions` 中配置：
   - `ALIYUN_CONTAINER_REGISTRY_USERNAME`
   - `ALIYUN_CONTAINER_REGISTRY_PASSWORD`
   - `ALIYUN_CONTAINER_REGISTRY_NAMESPACE`

## GitHub Actions 工作流配置

```yaml title=".github/workflows/docker-build-and-push.yml"
name: Docker Build and Push to Aliyun

on:
    push:
        tags:
            - 'v*' # 推送 v* 格式标签时触发

jobs:
    build-and-push:
        runs-on: ubuntu-latest
        steps:
            - name: Checkout Code
              uses: actions/checkout@v4
              with:
                  fetch-depth: 0

            - name: Login to Aliyun Container Registry
              run: |
                  echo "${{ secrets.ALIYUN_CONTAINER_REGISTRY_PASSWORD }}" | docker login \
                    --username "${{ secrets.ALIYUN_CONTAINER_REGISTRY_USERNAME }}" \
                    --password-stdin registry.cn-shanghai.aliyuncs.com

            - name: Build and Push Docker Image
              env:
                  IMAGE: registry.cn-shanghai.aliyuncs.com/${{ secrets.ALIYUN_CONTAINER_REGISTRY_NAMESPACE }}/yiwen-blog-website:${{ github.ref_name }}
              run: |
                  docker build --platform linux/amd64 -t "$IMAGE" .
                  docker push "$IMAGE"
```

## 说明

- 镜像版本直接采用触发构建的 Git Tag（`${{ github.ref_name }}`）动态标记，避免硬编码版本号。
- 指定 `--platform linux/amd64` 确保在云端 `x86_64` 服务器上正常运行。
