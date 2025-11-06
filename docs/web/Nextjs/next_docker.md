---
title: 使用 nginx 映射 next 资源的问题
sidebar_label: 使用 nginx 映射 next 资源的问题
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 使用 nginx 映射 next 资源的问题

## 介绍

nextjs 如果还用图片资源，框架自身会对图片资源进行优化，优化后的资源路径不再是原来的路径，而是`/_next/image`，如果使用nginx部署时候只给`/`路径做映射会导致图片无法加载问题，在生产部署时候需要注意这个问题。

## 解决方案

```bash
server {
    listen 80;
    listen [::]:80;
    server_name _; # 替换为您的域名

    location /_next/image {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location / {
        proxy_pass http://localhost:3000; # 代理到 Next.js 应用
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```