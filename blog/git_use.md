# 常用git命令及解释

## package-lock.json应不应该上传到代码仓库

package-lock.json 文件应该上传到代码仓库。这个文件锁定了项目依赖的确切版本，这样当其他开发者或者 CI/CD 系统克隆并安装项目时，会得到与你完全相同的依赖版本。

上传 package-lock.json 的好处：

-   一致性: 它确保每个开发者和每个部署都使用相同版本的依赖，减少了“在我机器上运行得很好”问题。

-   性能: 由于所有版本都已确定，npm 可以更有效地重用缓存，减少了需要解析的依赖树。

-   安全性: 锁定依赖的版本可以更容易地进行安全审核，并快速应对安全漏洞。

-   审计和追踪: 有了 package-lock.json，你可以更容易地审计项目依赖，并查明哪个版本在什么时候由谁添加。

然而，也有一些特定情况可能不需要这么做：

-   库项目: 如果你正在开发一个被其他项目依赖的库（而不是一个最终用户应用），有些人选择不上传 package-lock.json，因为这个锁文件不会对依赖这个库的项目产生影响。

-   多种环境: 如果项目需要在多种不同的环境中运行，并且依赖项可能因此而有所不同，那么 package-lock.json 可能会引入问题。

-   团队协作问题: 在某些团队工作流中，package-lock.json 可能会频繁地引发合并冲突。这通常是工作流管理不当的结果，但解决这个问题的一个暂时方法是不上传 package-lock.json。

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
