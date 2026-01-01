---
sidebar_position: 22
title: 拉取私有仓库配置指南
tags: [go, golang, GOPRIVATE, git, ssh, 私有仓库, CI/CD]
---

# Go 拉取私有仓库配置指南

这是一个非常经典且必须解决的问题。默认情况下，Go 的工具链（`go get`）会尝试去 Go 官方的公共代理（proxy.golang.org）拉取代码，且不带任何身份认证。

要拉取私有仓库（比如公司内部的 GitLab/GitHub 私有库），你需要打通**两个环节**：

1. **告诉 Go 工具链**："这个仓库是私有的，别走公共代理，直接去源站拉"。
2. **告诉 Git 工具**："我有权限，这是我的钥匙（SSH Key 或 Token）"。

---

## 第一步：配置 Go 环境变量 (GOPRIVATE)

这是最重要的一步。Go 默认启用 `GOPROXY`（代理）和 `GOSUMDB`（校验和数据库）。私有仓库既不存在于公共代理中，也无法通过公开的校验和验证，所以会报错。

你需要设置 `GOPRIVATE` 环境变量，告诉 Go 哪些域名下的仓库是私有的。

```bash
# 假设你们公司的私有仓库都在 gitlab.mycompany.com 下
go env -w GOPRIVATE=gitlab.mycompany.com/*

# 或者如果你们用的是 GitHub 私有库
go env -w GOPRIVATE=github.com/my_org/*
```

- **作用**：设置后，Go 对匹配该域名的依赖，会跳过 Proxy 和 SumDB，直接调用本地的 `git` 命令去源地址拉取。

---

## 第二步：配置 Git 鉴权

Go 底层是调用 `git` 命令拉取代码的。如果你的终端能用 `git clone` 拉下代码，`go get` 通常也能成功。

有两种主流方案：

### 方案 A：使用 SSH 协议（推荐，开发者本地最常用）

这是最方便的方法。前提是你已经生成了 SSH Key（`id_rsa.pub`）并添加到了 GitHub/GitLab 的个人设置里。

但是，Go 里的依赖路径通常写的是 `github.com/xxx`，Go 默认会尝试用 HTTPS (`https://github.com/xxx`) 去拉取，这需要输密码。

我们需要配置 Git，让它**遇到 HTTPS 时自动偷梁换柱改成 SSH**：

```bash
# 对于 GitHub
git config --global url."git@github.com:".insteadOf "https://github.com/"

# 对于公司内部 GitLab
git config --global url."git@gitlab.mycompany.com:".insteadOf "https://gitlab.mycompany.com/"
```

**配置完后：**
当 Go 执行 `git clone https://github.com/company/private-repo` 时，Git 会自动执行 `git clone git@github.com:company/private-repo`，利用你的 SSH Key 完成无感认证。

### 方案 B：使用 HTTPS + Access Token（CI/CD 流水线常用）

在 CI/CD（如 Jenkins, GitHub Actions）里，通常不方便配置 SSH Key，这时可以使用 **Personal Access Token**。

你需要创建一个 `~/.netrc` 文件（Linux/Mac）或 `_netrc` (Windows)，写入凭证：

```text
# ~/.netrc 文件内容
machine github.com
login <你的用户名>
password <你的Personal Access Token>

machine gitlab.mycompany.com
login <你的用户名>
password <你的Access Token>
```

配置好这个文件后，Git 使用 HTTPS 拉取时会自动读取这里的密码，无需人工干预。

---

## 总结操作流程

假设你要拉取 `github.com/my-company/proto-lib` 这个私有库：

1. **设置 GOPRIVATE:**

```bash
go env -w GOPRIVATE=github.com/my-company
```

2. **配置 Git 强制走 SSH (如果你本地有 SSH Key):**

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

3. **拉取代码:**

```bash
go get github.com/my-company/proto-lib
```

---

## 常见报错排查

- **403 Forbidden / Terminal prompts for password:**
  说明 Git 鉴权失败。检查 `git config --global -l` 看 `insteadOf` 配置对不对，或者检查你的 SSH Key 是否过期。

- **checksum mismatch / 404 Not Found:**
  说明 Go 还在走代理。检查 `go env GOPRIVATE` 是否覆盖了你的私有库域名。
