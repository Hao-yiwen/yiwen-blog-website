---
sidebar_position: 4
---

# 常用命令

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
