---
sidebar_position: 2
title: Go 常用命令速查手册
tags: [go, golang, commands, go mod, go build, go test]
---

# Go 常用命令速查手册

## 1. 项目初始化与模块管理 (Go Modules)

从 Go 1.11 开始，`go mod` 是管理依赖的标准方式。

| 命令 | 说明 | 常用场景 |
| --- | --- | --- |
| **`go mod init [模块名]`** | 初始化新模块 | 在项目根目录下创建 `go.mod` 文件。 |
| **`go mod tidy`** | **整理依赖 (最常用)** | 自动下载代码中 import 的包，并移除未使用的包。 |
| **`go mod download`** | 下载依赖到本地缓存 | 预先下载依赖，常用于 CI/CD 环境。 |
| **`go mod vendor`** | 导出依赖 | 将所有依赖项复制到项目下的 `vendor` 目录中。 |
| **`go mod graph`** | 打印模块依赖图 | 查看复杂的依赖关系树。 |
| **`go get [包名]`** | 添加/更新依赖 | `go get example.com/pkg@v1.2.3` (注意：不再用于安装可执行文件)。 |

> **小贴士：** 每次提交代码前，养成运行一次 `go mod tidy` 的习惯，保持 `go.mod` 和 `go.sum` 的整洁。

---

## 2. 代码开发与运行

日常编码中最频繁使用的命令。

### `go run`

编译并直接运行 Go 程序（不生成可执行文件）。

```bash
go run main.go
# 或者运行当前目录下所有文件
go run .
```

### `go fmt`

格式化代码。Go 语言强制统一代码风格。

```bash
# 格式化当前目录及所有子目录下的代码
go fmt ./...
```

### `go vet`

静态代码分析工具。它可以帮你发现代码中潜在的逻辑错误（如 printf 参数不匹配、不可达代码等）。

```bash
go vet ./...
```

---

## 3. 构建与安装

当代码准备好发布或部署时使用。

### `go build`

编译代码生成二进制可执行文件。

**基本用法：**

```bash
go build -o myapp main.go
```

`-o myapp` 指定输出文件名为 `myapp`

**交叉编译 (Cross Compilation)：**

Go 的杀手级功能，可以在一个平台上编译出另一个平台的可执行文件。

```bash
# 在 Mac/Linux 上编译 Windows 程序
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o app.exe

# 在 Windows (PowerShell) 上编译 Linux 程序
$env:CGO_ENABLED="0"; $env:GOOS="linux"; $env:GOARCH="amd64"; go build -o app
```

### `go install`

编译并安装工具。会将编译好的二进制文件移动到 `$GOPATH/bin` 目录下。

```bash
# 安装一个工具，例如 gopls 或 dlve
go install github.com/go-delve/delve/cmd/dlv@latest
```

---

## 4. 测试与性能分析

Go 原生自带强大的测试框架。

| 命令 | 说明 |
| --- | --- |
| **`go test`** | 运行当前目录下的测试。 |
| **`go test ./...`** | 运行当前及所有子目录的测试。 |
| **`go test -v`** | 显示详细的测试日志 (Verbose)。 |
| **`go test -run TestName`** | 仅运行名称匹配 `TestName` 的测试用例。 |
| **`go test -cover`** | 查看简易的代码覆盖率。 |
| **`go test -bench=.`** | 运行性能测试 (Benchmark)。 |
| **`go test -race`** | **竞态检测**。检测并发代码中是否存在数据竞争（非常重要）。 |

---

## 5. 环境配置与查看

### `go env`

查看或设置 Go 的环境变量。

**查看所有变量：**

```bash
go env
```

**设置国内代理 (中国开发者必备)：**

为了加速依赖包下载，建议配置 `GOPROXY`。

```bash
go env -w GOPROXY=https://goproxy.cn,direct
```

**开启 Go Modules：**

```bash
go env -w GO111MODULE=on
```

### `go version`

查看当前 Go 的版本。

```bash
go version
```

---

## 6. 文档与帮助

### `go doc`

在终端直接查看包或函数的文档，无需上网。

```bash
# 查看 fmt 包的文档
go doc fmt

# 查看 fmt.Println 函数的文档
go doc fmt.Println
```

### `go tool`

访问底层工具链（如汇编查看、pprof 分析等）。

```bash
# 查看汇编代码
go tool compile -S main.go
```

---

## 7. 开发工作流常用组合

1. **新项目：** `go mod init project_name`
2. **写代码：** `go run main.go` (调试)
3. **加依赖：** `go get github.com/gin-gonic/gin` -> `go mod tidy`
4. **提交前：** `go fmt ./...` -> `go vet ./...` -> `go test ./...`
5. **上线前：** `go build -o app main.go`

---

## 8. `go mod tidy` vs `go mod download` 详解

当你运行 `go mod tidy` 时，它为了计算依赖图和校验哈希值，确实会把缺少的包下载下来。乍一看 `go mod download` 似乎是多余的。

但实际上，它们设计的**目的（Intent）**和**依赖的"真相来源"**完全不同。

简单来说：

- **`go mod tidy`** 是为了 **"修剪"**：它看你的代码（`.go`文件），决定 `go.mod` 里该保留什么、删掉什么。
- **`go mod download`** 是为了 **"预热"**：它只看 `go.mod` 文件，不管代码怎么写，先把文件里列出的东西下载好。

### 核心区别：真相来源不同

| 特性 | `go mod tidy` | `go mod download` |
| --- | --- | --- |
| **参考依据** | **源代码 (`.go` 文件)** + `go.mod` | **仅 `go.mod` 和 `go.sum`** |
| **是否修改文件** | **是** (会修改 `go.mod` 和 `go.sum`) | **否** (只下载，不改文件) |
| **主要动作** | 添加缺少的包，**删除没用的包** | 机械地下载 `go.mod` 里列出的所有包 |
| **使用场景** | 日常开发、提交代码前 | CI/CD 流水线、Docker 构建 |

### 为什么还需要 `go mod download`？

#### 场景一：Docker 构建缓存 (最重要用途)

这是 `go mod download` 存在的最大意义。在编写 Dockerfile 时，我们希望利用 Docker 的层（Layer）缓存机制来加速构建。

**如果你只用 `go mod tidy`：**

你需要先把所有源代码 `COPY` 进去，才能运行 `tidy`。这意味着只要你改了一行代码（即使没改依赖），Docker 缓存就失效了，必须重新下载所有依赖。

**如果你用 `go mod download`：**

你可以先把 `go.mod` 和 `go.sum` 复制进去下载依赖。只要依赖没变，这一层缓存就一直有效，无论你后面怎么改代码，都不用重新下载包。

**最佳实践 Dockerfile 写法：**

```dockerfile
# 1. 先只复制依赖定义文件
COPY go.mod go.sum ./

# 2. 下载依赖 (这一步会被 Docker 缓存住！)
RUN go mod download

# 3. 依赖下好了，再复制源代码
COPY . .

# 4. 构建
RUN go build -o app .
```

#### 场景二：CI/CD 流水线 (纯净环境)

在持续集成（CI）环境中，通常希望环境是**可预测**的。

- 运行 `go mod tidy` 可能会修改 `go.mod`（例如你本地忘了 tidy 就提交了）。在 CI 里修改被版本控制的文件通常是不好的实践。
- 运行 `go mod download` 则是只读模式。它保证下载的依赖严格等于 `go.mod` 中锁定的版本，如果 `go.mod` 有问题，它不会自作聪明去修，而是直接报错或只下载指定的，保证了构建的一致性。

#### 场景三：离线开发准备

如果你要去一个没有网络的地方开发（比如飞机上），或者网络很差。

你可以先运行 `go mod download`。它会把 `go.mod` 里涉及的所有依赖全部下载到本地缓存（Module Cache）中。

而 `go mod tidy` 有时候发现代码里没用到某个包，可能就不会去下载它的完整内容，或者把它清理掉了。`download` 确保了清单里的东西都在本地。

### 总结

- **开发时：** 几乎只用 **`go mod tidy`**。你需要它来帮你自动管理 `import` 和 `go.mod` 的同步。
- **部署/构建时：** 必须用 **`go mod download`**。它是为了利用缓存加速构建，以及确保在不修改文件的情况下准备好依赖环境。
