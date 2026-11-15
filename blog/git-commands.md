---
title: Git 常用命令、SSH 配置与 Cherry-Pick 实战
authors: [yiwen]
tags: [Git, GitHub, SSH, 开发工具]
date: 2023-09-14
---

整理日常协作中高频使用的 Git 撤销命令、别名设置、SSH 密钥配置、锁文件管理规范，以及 `cherry-pick` 与 GUI 客户端实践。

<!-- truncate -->

## 常用撤销命令

```bash
# 丢弃工作区所有未暂存的修改
git checkout -- .
# 或使用新版命令
git restore .

# 丢弃暂存区与工作区的所有未提交更改
git reset --hard
```

## 设置常用别名

```bash
# git status -> git st
git config --global alias.st status

# git pull -> git pl
git config --global alias.pl pull

# git push -> git sh
git config --global alias.sh push
```

## 选择性合并：git cherry-pick 与 Fork 客户端

`git cherry-pick` 允许将一个或多个特定提交从其他分支精准复制到当前分支，而无需合并整条分支：

- **选择性合并**：只需要引入某个功能提交或线上热修补丁（Hotfix）时非常高效。
- **跨分支同步**：在维护多版本发布分支时，可将修复精准挑拣到目标分支。

```bash
# 将指定 commit 应用到当前分支
git cherry-pick <commit-hash>
```

在日常开发中，配合轻量流畅的 Git GUI 客户端 [Fork](https://git-fork.com/) 执行交互式 `cherry-pick`、查看提交图谱与解决冲突，体验非常顺手。

## 远程仓库切换 SSH 连接

由于国内环境下通过 `HTTPS` 连接 GitHub 偶发超时，推荐将远程地址切换为 `SSH` 协议：

```bash
# 1. 查看当前远程地址
git remote -v

# 2. 切换为 SSH 地址
git remote set-url origin git@github.com:USERNAME/REPOSITORY.git
```

### 配置 GitHub SSH 密钥

1. 检查本地是否已有密钥：

```bash
ls -al ~/.ssh
```

2. 生成新的 SSH 密钥对（推荐使用 `ed25519` 或 `rsa 4096`）：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

3. 启动 `ssh-agent` 并添加私钥：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

4. 查看公钥内容并添加到 GitHub 账户 `Settings -> SSH and GPG keys`：

```bash
cat ~/.ssh/id_ed25519.pub
```

5. 测试连接是否生效：

```bash
ssh -T git@github.com
```

## 锁文件（Lockfile）应不应该提交到仓库？

应用项目的锁文件（如 `pnpm-lock.yaml`、`package-lock.json`）**应当提交到代码仓库**，以锁定依赖树中所有子依赖的确切版本：

- **环境一致性**：确保每位开发者与 CI/CD 构建环境安装完全一致的依赖版本，避免“在我电脑上是好的”。
- **构建提速**：版本树已确定，包管理器可以直接复用缓存并跳过复杂的版本解析。
- **安全可审计**：便于追踪某个传递依赖是在哪次提交中引入或升级的。

注：只有纯基础库（Library）项目或需要避免混用多包管理器的场景才需特殊处理；并且同一仓库应当只保留一种包管理器的锁文件。
