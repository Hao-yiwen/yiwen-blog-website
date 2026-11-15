---
title: 开发环境常用命令速查
authors: [yiwen]
tags: [终端, macOS, npm, Homebrew]
date: 2023-09-13
---

整理日常开发中高频使用的终端代理、镜像源配置、Homebrew 服务管理与 macOS 网络配置命令。

<!-- truncate -->

## 终端临时代理

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

## 常用包管理器镜像源

### npm / pnpm / yarn

```bash
# npmmirror（原淘宝镜像新域名）
npm config set registry https://registry.npmmirror.com
pnpm config set registry https://registry.npmmirror.com
yarn config set registry https://registry.npmmirror.com

# 恢复 npm 官方源
npm config set registry https://registry.npmjs.org/
```

### pip

修改 `~/.pip/pip.conf`：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

### Docker

修改 `/etc/docker/daemon.json`：

```json
{
    "registry-mirrors": ["https://your-mirror.com"]
}
```

## Homebrew 常用命令

```bash
# 查看软件包信息
brew info nginx

# 重启后台服务
brew services restart nginx

# 查看服务状态
brew services info nginx
```

## macOS 修改 Hosts

```bash
# 编辑 hosts 文件
sudo vim /etc/hosts

# 示例映射
# 127.0.0.1 example.com

# 刷新 DNS 缓存
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```
