---
title: GitHub Actions 自动化构建与部署：从 0 到 1
authors: [yiwen]
tags: [GitHub Actions, GitHub Pages, Docker, CI/CD]
date: 2023-09-09
---

介绍如何使用 GitHub Actions 结合 GitHub Pages 与容器镜像仓库，为前端静态站点实现从代码提交到自动构建发布的完整 CI/CD 流水线。

<!-- truncate -->

## 前提条件

1. 一个 GitHub 账号与托管在 GitHub 上的前端项目。
2. 基本的 Node.js 与 Docker 构建知识。

## 一、GitHub Pages 自动化发布工作流

GitHub Actions 基于仓库下 `.github/workflows/*.yml` 文件声明触发规则与执行步骤。结合 `peaceiris/actions-gh-pages`，可以在推送到 `master` 分支时自动构建并部署静态站点：

```yaml title=".github/workflows/deploy.yml"
name: Deploy to GitHub Pages

on:
    push:
        branches: [master]

permissions:
    contents: write

jobs:
    deploy:
        runs-on: ubuntu-latest
        steps:
            - name: Checkout
              uses: actions/checkout@v4
              with:
                  fetch-depth: 0

            - name: Setup pnpm
              uses: pnpm/action-setup@v4

            - name: Setup Node.js
              uses: actions/setup-node@v4
              with:
                  node-version: 22
                  cache: pnpm

            - name: Install and Build
              run: |
                  pnpm install --frozen-lockfile
                  pnpm build

            - name: Deploy to GitHub Pages
              uses: peaceiris/actions-gh-pages@v4
              with:
                  github_token: ${{ secrets.GITHUB_TOKEN }}
                  publish_dir: ./build
                  publish_branch: gh-pages
```

### 核心步骤说明

1. **Checkout 代码**：拉取仓库源码，`fetch-depth: 0` 拉取完整 Git 历史以支持生成文档最后更新时间。
2. **安装 Node.js 与 pnpm**：自动读取 `package.json` 中的 `packageManager` 版本并开启依赖缓存。
3. **构建静态产物**：运行 `pnpm build` 将生产文件输出到 `./build`。
4. **发布到 `gh-pages` 分支**：利用内置 `GITHUB_TOKEN`（配合 `contents: write` 权限）自动推送到 GitHub Pages 分支。

## 二、结合 Docker 镜像构建推送

如果需要同时打包容器镜像用于自建服务器部署，可以在推送版本 Tag 时构建并推送镜像：

```yaml title="Docker 构建与推送片段"
- name: Login to Container Registry
  run: echo "${{ secrets.REGISTRY_TOKEN }}" | docker login -u "${{ secrets.REGISTRY_USER }}" --password-stdin

- name: Build and Push Image
  run: |
      docker build --platform linux/amd64 -t ${{ secrets.REGISTRY_USER }}/yiwen-blog-website:latest .
      docker push ${{ secrets.REGISTRY_USER }}/yiwen-blog-website:latest
```

## 故障排查

1. **查看构建日志**：在仓库 `Actions` 选项卡中点开失败的 Job 查看具体报错阶段。
2. **检查仓库权限**：确保 `Settings -> Pages` 已将 Source 指向 `gh-pages` 分支根目录，且 Workflow 具备 `contents: write` 权限。
