---
title: Nginx 核心配置与 root / alias 区别详解
authors: [yiwen]
tags: [Nginx, 运维, 服务端]
date: 2023-09-20
---

梳理 Nginx 中静态资源映射最易混淆的 `root` 与 `alias` 指令区别，并附一份常用的反向代理与静态资源服务配置示例。

<!-- truncate -->

## `root` 与 `alias` 的区别

### `root`（拼接完整 URI）

使用 `root` 时，请求的完整 URI（包含 `location` 匹配前缀）会直接拼接到 `root` 指定的目录之后，即：`文件路径 = root + location + 剩余相对路径`。

```nginx
location /images/ {
    root /data;
}
```

访问 `/images/pic.jpg` 时，实际读取的文件路径为 `/data/images/pic.jpg`。

### `alias`（替换匹配前缀）

使用 `alias` 时，`location` 匹配到的路径前缀会被**替换**为 `alias` 指定的目录（注意结尾斜杠保持一致）：

```nginx
location /images/ {
    alias /data/photos/;
}
```

访问 `/images/pic.jpg` 时，实际读取的文件路径为 `/data/photos/pic.jpg`。

## Nginx 完整配置示例

```nginx title="nginx.conf"
worker_processes  1;

events {
    worker_connections  1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;

    sendfile        on;
    keepalive_timeout  65;

    # 后端文件服务上游节点
    upstream fileserver {
        server 127.0.0.1:9000 weight=10;
    }

    server {
        listen 8090;
        server_name file.example.com;
        ssi on;
        ssi_silent_errors on;

        location /video {
            proxy_pass http://fileserver;
        }

        location /mediafiles {
            proxy_pass http://fileserver;
        }
    }

    # 静态站点与跨域配置示例
    server {
        listen       8090;
        server_name  localhost www.example.com;
        ssi on;
        ssi_silent_errors on;

        location / {
            alias /var/www/static-portal/;
            index index.html index.htm;
        }

        location /static/img/ {
            alias /var/www/static-portal/img/;
        }

        location /static/css/ {
            alias /var/www/static-portal/css/;
        }

        location /static/js/ {
            alias /var/www/static-portal/js/;
        }

        location /static/plugins/ {
            alias /var/www/static-portal/plugins/;
            add_header Access-Control-Allow-Origin http://ucenter.example.com;
            add_header Access-Control-Allow-Credentials true;
            add_header Access-Control-Allow-Methods GET;
        }

        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
    }

    include servers/*;
}
```
