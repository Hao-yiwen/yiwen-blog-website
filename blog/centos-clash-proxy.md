---
title: CentOS 服务器配置 Clash 与 Docker 代理
authors: [yiwen]
tags: [CentOS, Linux, Docker, 网络代理]
date: 2023-11-23
---

在 Linux 服务器部署服务或构建镜像时，经常因无法稳定访问 GitHub 或 Docker Hub 导致拉取超时。本文记录在 CentOS 上配置 Clash 守护进程以及为 Docker Daemon 配置 HTTP 代理的完整步骤。

<!-- truncate -->

## 一、部署 Clash 守护进程

```bash
# 1. 解压二进制包（解压后生成同名无 .gz 后缀文件）
gzip -d clash-linux-amd64-latest.gz
mv clash-linux-amd64-latest clash
chmod +x clash

# 2. 将配置文件 config.yaml 与 Country.mmdb 放入 /root/.config/clash/
mkdir -p /root/.config/clash
# 先在前台试运行确认无报错
./clash -d /root/.config/clash

# 3. 移动到系统路径
cp clash /usr/local/bin/
```

创建 `systemd` 服务文件 `/etc/systemd/system/clash.service`：

```ini title="/etc/systemd/system/clash.service"
[Unit]
Description=Clash daemon, A rule-based proxy in Go.
After=network.target

[Service]
Type=simple
Restart=always
ExecStart=/usr/local/bin/clash -d /root/.config/clash

[Install]
WantedBy=multi-user.target
```

启动并设为开机自启：

```bash
systemctl daemon-reload
systemctl enable --now clash
systemctl status clash
```

终端临时开启代理并测试：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
curl -I https://www.google.com
```

## 二、为 Docker Daemon 配置代理

注意：终端的 `http_proxy` 环境变量对 `docker pull` **不生效**，因为镜像拉取是由后台 `dockerd` 守护进程发起的。遇到如下超时报错时：

```text
Get https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

需要为 `docker.service` 单独配置环境变量：

1. 创建目录并编辑 `/etc/systemd/system/docker.service.d/http-proxy.conf`：

```ini title="/etc/systemd/system/docker.service.d/http-proxy.conf"
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
Environment="NO_PROXY=localhost,127.0.0.1,*.aliyuncs.com"
```

2. 重载配置并重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

参考：[Docker Daemon Proxy 官方文档](https://docs.docker.com/engine/daemon/proxy/)
