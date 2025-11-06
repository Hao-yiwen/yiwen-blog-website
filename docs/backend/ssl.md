---
title: nginx配置ssl证书
sidebar_label: nginx配置ssl证书
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# nginx配置ssl证书

## 生成CSR

因为我使用的是阿里云服务，可以自动生成`csr`，`csr`在证书网站要配置，私钥要保存好放在服务器中。

## 证书购买

找到一家价格相对便宜的`dv`级别的`ssl证书`网站，价格相对便宜，一年差不多11美金。[cheapsslsecurity](https://cheapsslsecurity.com/)

## 上传证书

申请好`ssl证书`后下载证书到本地，然后上传证书到服务器某为止。

## nginx配置

```yml
server {
        listen 443 ssl;
        server_name _;

        ssl_certificate "/opt/xxx.crt";  # 配置证书在服务器路径
        ssl_certificate_key "/opt/xxx.key"; # 配置私钥在服务器路径
        ssl_session_cache shared:SSL:1m;
        ssl_session_timeout 10m;
        ssl_ciphers PROFILE=SYSTEM;
        ssl_prefer_server_ciphers on;

        location / {
            proxy_pass http://localhost:3000; 
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
        }

    }
```