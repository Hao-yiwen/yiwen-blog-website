---
title: 常用命令
date: 2023-09-13
---

# 常用命令

<!-- truncate -->

## 终端代理
```bash title="ss"
export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;
```

## 常用命令

1. 设置镜像源

```bash
淘宝: npm config set registry https://registry.npm.taobao.org

npm官方: npm config set registry https://registry.npmjs.org/

yarn: yarn config set registry https://registry.npm.taobao.org

pip:
# 修改 ~/.pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple

docker:
# 修改 /etc/docker/daemon.json
{
  "registry-mirrors": ["https://your-mirror.com"]
}
```

2. 常用brew命令

```bash
// 包信息
brew info nginx
// 重启服务
brew services restart nginx
// 服务信息
brew services info nginx
```

3. mac修改hosts

```bash
// 编辑hosts
sudo vim /etc/hosts
// 添加网址
127.0.0.1    example.com
// 刷新 DNS 缓存
sudo dscacheutil -flushcache
```
