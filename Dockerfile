# 阶段 1：构建静态站点
FROM node:22-alpine AS builder

# git：文档「最后更新时间」需要读取 git 历史（Docusaurus Faster 在无 git 时会构建失败）
RUN apk add --no-cache git && npm install -g pnpm@11.9.0

# 镜像内不需要安装 git hooks
ENV HUSKY=0

WORKDIR /app

# 先只复制依赖清单，利用 Docker 层缓存
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

# 阶段 2：使用 Nginx 托管构建产物
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
