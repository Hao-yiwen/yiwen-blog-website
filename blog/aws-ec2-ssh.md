---
title: AWS EC2 实例 SSH 密钥登录配置
authors: [yiwen]
tags: [AWS, SSH, Linux, 运维]
date: 2024-01-13
---

购买 AWS EC2 服务器后，初次连接时发现无法像国内常见云厂商那样直接用密码登录。研究后发现，EC2 默认要求使用 SSH 密钥对登录；如果在创建实例时未绑定密钥对，则需要手动配置公钥。

<!-- truncate -->

## 配置流程

```bash
# 1. 本地生成 SSH 密钥对
ssh-keygen -t rsa -b 2048 -f ~/my-aws-key

# 2. 查看并复制公钥内容
cat ~/my-aws-key.pub

# 3. 在 EC2 实例中将公钥写入 authorized_keys
vim ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 4. 本地赋予私钥安全权限并登录
chmod 400 ~/my-aws-key
ssh -i ~/my-aws-key ec2-user@instance-ip
```

## 参考链接

- [AWS EC2 控制台（东京区域）](https://ap-northeast-1.console.aws.amazon.com/console/home?region=ap-northeast-1#)
- [站点搭建参考视频](https://www.youtube.com/watch?v=G38pR9nYYL8)
- [Cloudflare 配置参考](https://www.moeelf.com/archives/304.html)
