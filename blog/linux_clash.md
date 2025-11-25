---
title: Centos代理
date: 2023-11-23
---

# Centos代理

在服务器使用过程中我们会经常发现各类问题，但是这类问题都有一个公共问题，就是很多时候没法正常访问`github`资源或者`google`资源，从而导致部署卡壳。那么如何处理此类问题就显得尤为重要，在这里我介绍一种我使用的`clash + centos`的解决方案。

<!-- truncate -->

## clash下载

:::tip
安装 `clash-linux-amd64-latest.gz`
:::
[谷歌云盘下载](https://drive.google.com/drive/folders/1mhKMWAcS5661t_TWSp9wm4WNj32NFbZK)

## 安装

```bash
# 解压压缩文件，会生成一个没有 .gz 的同名文件
gzip -d clash-linux-amd64-latest.gz

# （可选项）修改程序文件名
mv clash-linux-amd64-latest.gz clash

# 添加运行权限
chmod +x clash

# 先运行服务
# 在运行钱请将自己的 config.yaml 复制到 /root/.config/clash 下面
# 如果有Country.mmdb报错 那可能需要 wget -O Country.mmdb https://www.sub-speeder.com/client-download/Country.mmdb 解决
./clash

//添加守护进程
cp clash /usr/local/bin

// 添加配置
vim /etc/systemd/system/clash.service

# start
[Unit]
Description=Clash daemon, A rule-based proxy in Go.
After=network.target

[Service]
Type=simple
Restart=always
ExecStart=/usr/local/bin/clash -d /root/.config/clash

[Install]
WantedBy=multi-user.target
# end

# 系统启动时运行
systemctl enable clash

# 立即运行 clash 服务
systemctl start clash

# 查看 clash 服务运行状态
systemctl status clash

# 查看运行日志
journalctl -xe

# 使用时临时修改
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 测试
curl https://www.google.com
```

## 魔法使用完毕

## docker中使用代理

docker hub在国内使用的时候会有几率出现链接超时问题，这时需要给docker设置代理来解决这个问题。

```bash
Get https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
```

### 解决方案

https://stackoverflow.com/questions/48056365/error-get-https-registry-1-docker-io-v2-net-http-request-canceled-while-b

https://docs.docker.com/engine/daemon/proxy/

从两者来看需要手动设置代理

1. 创建或者编辑`/etc/systemd/system/docker.service.d/http-proxy.conf`
```bash
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
```

2. 重启
```bash
sudo systemctl daemon-reload                            
sudo systemctl restart docker
```