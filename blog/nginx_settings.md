---
title: nginx配置
date: 2023-09-20
---

# nginx配置

<!-- truncate -->

## root和alias区别

### root

当使用 root 时，请求的完整 URI（从 location 部分开始）会被直接附加到 root 指令指定的路径上。这意味着 root + location + 请求的相对 URI = 文件系统上的绝对路径。

```bash
location /images/ {
    root /data;
}
```

请求 /images/pic.jpg 对应的文件路径就是 /data/images/pic.jpg。

### alias

**使用 alias 时，location 中定义的路径会被 alias 指定的路径替换。**

```bash
location /images/ {
    alias /data/photos/;
}
```

请求 /images/pic.jpg 对应的文件路径就是 /data/photos/pic.jpg。

## nginx示例

```bash
# 设置工作进程的数量。
worker_processes  1;

# 设置每个工作进程的最大连接数。
events {
    worker_connections  1024;
}

# HTTP 配置块。
http {
    # 包含 MIME 类型定义。
    include       mime.types;
    # 默认 MIME 类型。
    default_type  application/octet-stream;

    # 开启高效文件传输模式。
    sendfile        on;

    # 设置 keep-alive 超时时间。
    keepalive_timeout  65;

    # 文件服务
    upstream fileserver {
        server 127.0.0.1:9000 weight=10;
    }

    server {
        listen 8090;
        server_name file.51xuecheng.cn;
        #charset koi8-r;
        ssi on;
        ssi_silent_errors on;

        location /video {
            proxy_pass http://fileserver
        }

        location /mediafiles {
            proxy_pass http://fileserver
        }

    }

    # 配置 HTTP 服务器。
    server {
        # 设置监听的端口和服务器名称。
        listen       8090;
        server_name  localhost www.51xuecheng.cn;
        #charset koi8-r;
        ssi on;
        ssi_silent_errors on;

        # 配置对根 URL 的请求。
        location / {
            # 设置目录别名。
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/;
            # 设置默认索引文件。
            index index.html index.htm;
        }

        # 配置静态资源（img）。
        location /static/img/ {
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/img/;
        }

        # 配置静态资源（css）。
        location /static/css/ {
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/css/;
        }

        # 配置静态资源（js）。
        location /static/js/ {
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/js/;
        }

        # 配置静态资源（plugins）并设置 CORS 头部。
        location /static/plugins/ {
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/plugins/;
            add_header Access-Control-Allow-Origin http://ucenter.51xuecheng.cn;
            add_header Access-Control-Allow-Credentials true;
            add_header Access-Control-Allow-Methods GET;
        }

        # 配置插件资源。
        location /plugins/ {
            alias /Users/yw.hao/Downloads/xc-ui-pc-static-portal/plugins/;
        }

        # 配置错误页面。
        error_page   500 502 503 504  /50x.html;

        # 重定向到错误页面。
        location = /50x.html {
            root   html;
        }
    }

    # 包括其他服务器配置。
    include servers/*;
}
```
