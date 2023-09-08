# 使用 Nginx 官方镜像作为基础镜像
FROM nginx:alpine

# 将 Docusaurus build 输出目录的内容复制到 Nginx 镜像中
COPY ./build /usr/share/nginx/html

# 暴露 80 端口
EXPOSE 80

# 运行 Nginx
CMD ["nginx", "-g", "daemon off;"]