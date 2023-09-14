---
sidebar_position: 6
---

# 常用git命令及解释

## git rebase

用来合并多个分支或者深层次代码冲突修复。很好用但是使用起来比较复杂的一个命令。

## 查看当前仓库远程连接

```bash
git remote -v
```

以为国内使用`https`连接`github`经常失败，所以将`https`url改为`ssh`。

1. 打开终端，并进入到你的 Git 仓库的本地目录。运行以下命令以切换到 SSH：

```bash
git remote set-url origin YOUR_SSH_URL_HERE
```

2. 验证更改

```bash
git remote -v
```

## 代码仓库配置ssh

1. 检查现有的 SSH 密钥

```bash
ls -al ~/.ssh
```

2. 生成新的 SSH 密钥

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

3. 将 SSH 密钥添加到 ssh-agent

```bash
eval "$(ssh-agent -s)"

# 然后添加你的 SSH 密钥到 ssh-agent：

ssh-add ~/.ssh/id_rsa
```

4. 将公钥添加到 Git 账户

`Linux/macOS`: `cat ~/.ssh/id_rsa.pub`

复制公钥内容（以 ssh-rsa ... 开头，以你的电子邮件地址结束）。

然后，进入你的 Git 仓库托管服务（如 GitHub、GitLab 等），找到添加 SSH 密钥的选项，并粘贴你的公钥。

5. 测试 SSH 连接

```bash
ssh -T git@github.com
```

你应该会看到一个确认消息，表明 SSH 已成功配置。
