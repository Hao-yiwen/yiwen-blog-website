# Yiwen's Blog

欢迎来到我的个人博客！这是一个基于 Docusaurus 构建的技术博客网站，记录我在软件开发和人工智能领域的学习与实践。

在线访问：https://hao-yiwen.github.io/yiwen-blog-website/

## 简介

这是我的个人技术博客，主要内容包括：

- **人工智能** - 深度学习、Transformer、PyTorch、Hugging Face、AI 工程化等
- **Native 开发** - iOS、Android、React Native、Flutter、HarmonyOS 等
- **Web 开发** - React、Next.js、CSS、JavaScript 等前端技术
- **服务端开发** - Go、Java、MySQL、Redis、Kubernetes 等后端技术
- **计算机基础** - 算法、设计模式、网络等基础知识
- **数学** - 概率论等

## 技术栈

- [Docusaurus 3](https://docusaurus.io/)（启用 Docusaurus Faster：Rspack + SWC）
- React 19 + TypeScript 5
- KaTeX 数学公式、Mermaid 图表
- Giscus 评论（GitHub Discussions）
- GitHub Pages 部署

## 环境要求

- Node.js >= 22.22.2（推荐使用 Node.js 22 LTS）
- 包管理器：[pnpm](https://pnpm.io/) 12（版本以 `package.json` 的 `packageManager` 字段为准；可用 Corepack：`corepack enable`）

## 本地开发

```bash
# 启用 Corepack 并安装依赖（同时会安装 git hooks）
corepack enable
pnpm install

# 启动开发服务器（默认仅 zh locale，http://localhost:3000）
pnpm start

# 生产构建 / 本地预览构建产物
pnpm build
pnpm serve

# 仅构建中文 locale（本地验证更快）
pnpm build --locale zh

# 类型检查 / CSS 检查
pnpm typecheck
pnpm lint:css
```

## 部署说明

- 推送到 `master` 后，GitHub Actions 会构建并通过 `gh-pages` 分支发布到 GitHub Pages
- 推送 `v*` 标签时，会构建 Docker 镜像并推送到阿里云容器镜像服务

## 许可证

MIT

## 联系方式

- GitHub: [Hao-yiwen](https://github.com/Hao-yiwen)
