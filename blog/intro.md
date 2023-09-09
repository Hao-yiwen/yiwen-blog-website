---
sidebar_position: 1
---

# GitHub 自动化部署：从0到1

本指南旨在引导您使用 GitHub Actions 和 GitHub Pages，高效地为前端项目实现自动化部署。自动化部署不仅可以提高开发效率，还能确保代码质量。

## TODO

github action 连接服务器，进行 docker 部署。

## 前提条件

1. 一个 GitHub 账号。
2. 基本的 Docker 知识。3.一个已上传至 GitHub 的 Node.js 项目

## 一、GitHub Actions：持续集成与部署工具

GitHub Actions 是一个用于持续集成和持续部署（CI/CD）的自动化工具。其基于 YML 文件进行配置，并能轻松集成至任何 GitHub 项目。

### 示例配置代码

```yml title=".github/workflows/docker-build.yml"
name: Docker Build and Push

on:
    push:
        branches:
            - master # Or the branch name you want

jobs:
    build:
        runs-on: ubuntu-latest

        steps:
            - name: Checkout Code
              uses: actions/checkout@v2

            - name: Setup Node.js environment
              uses: actions/setup-node@v2
              with:
                  node-version: '16'

            - name: Cache Node.js modules
              uses: actions/cache@v2
              with:
                  path: ~/.npm
                  key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
                  restore-keys: |
                      ${{ runner.os }}-node-

            - name: Install Dependencies
              run: npm install

            - name: Build Project
              run: npm run build

            - name: Login to Docker Hub
              run: echo "${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}" | docker login -u "${{ secrets.DOCKER_HUB_USERNAME }}" --password-stdin

            - name: Build Docker Image
              run: docker build --platform linux/amd64 -t ${{ secrets.DOCKER_HUB_USERNAME }}/yiwen-blog-website:latest .

            # githubpages
            - name: Push to Docker Hub
              run: docker push ${{ secrets.DOCKER_HUB_USERNAME }}/yiwen-blog-website:latest

            - name: Deploy to GitHub Pages
              uses: peaceiris/actions-gh-pages@v3
              with:
                  github_token: ${{ secrets.PERSONAL_ACCESS_TOKEN }} # 使用 GitHub Token 授权
                  publish_dir: ./build # 指定要部署的静态文件目录
                  publish_branch: gh-pages # 指定要发布的分支
```

## 步骤解释：

1. Checkout 代码 - 从 GitHub 仓库中拉取代码。
2. 设置 Node.js 环境 - 为项目构建和测试设置 Node.js 环境。
3. 缓存 Node.js 模块 - 加速后续的构建过程。
4. 安装依赖 - 使用 npm install 安装项目依赖。
5. 项目打包 - 使用 npm run build 打包项目。
6. 登录 Docker Hub - 使用提供的密钥登录 Docker Hub。
7. 构建并推送 Docker 镜像 - 创建 Docker 镜像并推送到 Docker Hub。
8. 部署到 GitHub Pages - 将构建的静态文件部署到 GitHub Pages。

## 二、GitHub Pages —— 静态网站托管

GitHub Pages 允许您直接从 GitHub 仓库托管静态网站。结合 GitHub Actions，静态网站的部署可以完全自动化。

## 示例代码

```yml
 - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
            github_token: ${{ secrets.PERSONAL_ACCESS_TOKEN }} # 使用 GitHub Token 授权
            publish_dir: ./build # 指定要部署的静态文件目录
            publish_branch: gh-pages # 指定要发布的分支
```

## 设置步骤：

1. 在 GitHub 仓库设置中，启用 GitHub Pages。
2. 选择一个用于部署的分支和路径。
3. 使用 peaceiris/actions-gh-pages@v3 来部署。确保部署分支和配置分支相同。

## 故障排查

1. 查看构建日志：在 Actions 选项卡下，可以查看构建日志以定位问题。
2. 检查密钥和权限：确保提供的 GitHub 和 Docker Hub 的密钥是有效的，并具有适当的权限。

# 总结

通过 GitHub Actions 和 GitHub Pages，我们能够轻松实现前端项目的自动化部署。这不仅提高了工作效率，也有助于维护代码质量。无论您是独立开发者还是团队成员，这样的自动化设置都将大大有益。
