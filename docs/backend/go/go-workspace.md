---
sidebar_position: 23
title: Go Workspace 多模块开发指南
tags: [go, golang, workspace, go-work, 多模块, 本地开发, gRPC]
---

# Go Workspace 多模块开发指南

`go work` (Go Workspace) 正是为了解决本地多模块联调问题而生的。

使用 `go work` 后，你可以在本地同时编辑多个 Module（比如一个存放 proto 的库，一个存放业务逻辑的服务），让它们**直接引用本地硬盘上的代码**，完全不需要推送到 Git，也不需要手动修改 `go.mod` 去写繁琐的 `replace` 指令。

---

## 1. 核心原理：它是如何工作的？

在 Go 1.18 之前，如果你在开发 `project-b` 时需要引用本地正在修改的 `project-a`，你必须在 `project-b/go.mod` 里写一句 `replace github.com/user/project-a => ../project-a`。这很麻烦，因为提交代码前你还得记得把它删掉。

**`go work` 的机制是：**
它在这些 Module 的上层创建一个 `go.work` 文件。当你在该目录下运行 Go 命令时，Go 编译器会忽略各个 `go.mod` 中的版本锁定，而是**优先在 `go.work` 声明的本地目录中查找依赖**。

---

## 2. 实战示例：模拟 gRPC 本地开发

假设你现在的目录结构是这样的（模拟微服务开发）：

```text
my-workspace/          <-- 工作区根目录
├── common-proto/      <-- 公共库 (存放 .proto 和生成的 .pb.go)
│   ├── go.mod         (module name: github.com/my/proto)
│   └── hello.pb.go
└── order-service/     <-- 业务服务 (依赖上面的 common-proto)
    ├── go.mod         (module name: github.com/my/order)
    └── main.go
```

### 第一步：初始化各个 Module

如果还没初始化，先确保它们是正常的 Go Module：

**common-proto/go.mod:**

```go
module github.com/my/proto

go 1.22
```

**order-service/main.go:**

```go
package main

import (
    "fmt"
    // 注意：这里我们直接 import 那个私有库的路径
    // 哪怕 git 上根本没有这个库，或者 git 上是旧版本
    pb "github.com/my/proto"
)

func main() {
    fmt.Println("引用本地库成功")
}
```

### 第二步：初始化工作区 (关键步骤)

回到根目录 `my-workspace/`，执行以下命令：

1. **初始化 `go.work` 文件**

```bash
go work init
```

2. **将两个项目加入工作区**

```bash
go work use ./common-proto
go work use ./order-service
```

*(或者直接 `go work use -r .` 递归查找所有模块)*

此时，根目录下会生成一个 `go.work` 文件，内容如下：

```go
go 1.22

use (
    ./common-proto
    ./order-service
)
```

### 第三步：见证奇迹

现在，你在 `my-workspace` 的任何子目录下（比如 `order-service` 里）运行代码：

```bash
cd order-service
go run main.go
```

**Go 会自动检测到上层的 `go.work`，发现 `github.com/my/proto` 这个包在 `../common-proto` 目录下有一份本地代码，它会直接使用本地的那份代码，而不会去尝试连接 GitHub。**

---

## 3. `go work` 的三大优势

1. **无需 Git 即可联调**：
   你可以在 `common-proto` 里改一个字段，保存一下，然后在 `order-service` 里立刻就能用，完全不需要 commit、push、pull 这一套漫长的流程。

2. **保持 `go.mod` 纯净**：
   你不需要在 `order-service/go.mod` 里写 `replace`。这意味着当你开发完成，把代码推送到 Git 时，**不需要对 `go.mod` 做任何回滚或修改**。CI/CD 系统在服务器上跑的时候（因为没有 `go.work` 文件），它会照常去拉取 Git 上的依赖。

3. **跨项目重构神器**：
   如果你使用的 IDE 是 GoLand 或 VS Code，开启 `go work` 后，IDE 会把这几个文件夹视为一个"大项目"。你可以直接从 `order-service` 的代码 **跳转定义** 到 `common-proto` 的源码中，甚至跨项目重命名变量。

---

## 4. 注意事项（必看）

### 不要提交 `go.work` 到 Git

`go.work` 是你个人的本地开发环境配置。你的同事可能有不同的目录结构。

**强烈建议**在根目录的 `.gitignore` 中添加：

```text
go.work
go.work.sum
```

### 生产环境依赖

虽然本地开发爽了，但当你最终要把 `order-service` 上线时，你的 `common-proto` **必须**已经推送到了 Git 仓库（或者是私有代理），因为线上的构建机器通常不会用到 `go.work`，它还是会依照 `go.mod` 去拉取远程代码。

---

## 总结

对于 gRPC 场景，`go work` 是**最佳实践**：

1. 在本地建一个大文件夹。
2. 把 `proto` 项目放进去，把 `client` 项目放进去，把 `server` 项目放进去。
3. `go work init` 然后 `go work use ...` 全部加进去。
4. **开发体验就像在一个单体应用里写代码一样顺滑。**
