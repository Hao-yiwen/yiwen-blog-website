# 阶段 1: 构建项目
# 使用 Node 官方镜像作为基础镜像
FROM node:16-alpine as builder

# 设置工作目录
WORKDIR /app

# 复制 package.json 和 yarn.lock 文件
COPY package.json yarn.lock ./

# 安装依赖
RUN yarn install --frozen-lockfile

# 复制项目文件
COPY . .

# 构建项目
RUN yarn build

# 阶段 2: 部署到 Nginx
# 使用 Nginx 官方镜像作为基础镜像
FROM nginx:alpine

# 从构建器阶段复制构建的文件到 Nginx 镜像中
COPY --from=builder /app/build /usr/share/nginx/html

# 暴露 80 端口
EXPOSE 80

# 运行 Nginx
CMD ["nginx", "-g", "daemon off;"]
