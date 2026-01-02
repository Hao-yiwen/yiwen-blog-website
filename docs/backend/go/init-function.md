---
sidebar_position: 13
title: init函数与包初始化
tags: [go, init, 初始化, 包, 执行顺序]
---

# Go 语言 `init` 函数详解

`init` 函数是 Go 中一个非常特殊的函数：你永远不需要（也不能）手动调用它，它是**由 Go 运行时（Runtime）自动调用**的。

你可以把它理解为 **"包（Package）的构造函数"**。

---

## 1. 它是干嘛的？

它的主要作用是在程序真正开始执行逻辑（也就是 `main` 函数）**之前**，做一些**初始化准备工作**。

常见用途：
* 初始化复杂的变量（简单的可以用 `=` 直接赋值，复杂的逻辑得写在函数里）
* 检查配置文件或环境变量是否存在
* 注册数据库驱动（这是最常见的用法）

---

## 2. 执行顺序（必考点）

当程序启动时，执行顺序是这样的：

1. **导入包（Imports）：** 先去初始化你 import 的那些包
2. **包级常量/变量（Const/Var）：** 初始化当前包里的全局变量
3. **`init()` 函数：** 自动执行 `init` 函数
4. **`main()` 函数：** 最后才轮到 `main`

**图解顺序：**

```
import -> const/var -> init() -> main()
```

---

## 3. 代码演示

```go
package main

import "fmt"

// 1. 全局变量最先初始化
var GlobalVar = func() int {
    fmt.Println("1. 全局变量初始化...")
    return 100
}()

// 2. init 函数随后自动执行
func init() {
    fmt.Println("2. init 函数执行...")
    // 这里可以修改全局变量，或者做检查
    if GlobalVar == 100 {
        GlobalVar = 200 // 修改变量状态
    }
}

// 3. main 函数最后执行
func main() {
    fmt.Println("3. main 函数开始执行...")
    fmt.Printf("GlobalVar 的值是: %d\n", GlobalVar)
}
```

**运行结果：**

```text
1. 全局变量初始化...
2. init 函数执行...
3. main 函数开始执行...
GlobalVar 的值是: 200
```

---

## 4. `init` 的四大特性

| 特性 | 说明 |
| --- | --- |
| **无参无返回值** | 签名必须是 `func init() {}` |
| **不能被调用** | 不能在代码里写 `init()`，会报错。只能由 Go 系统自动调用 |
| **可以有多个** | 同一个文件或同一个包的不同文件里可以有多个 `init`，按文件名顺序执行 |
| **每个包只执行一次** | 即使这个包被很多地方 import 了，它的 `init` 也只会跑一次 |

---

## 5. 最经典的实战场景：`_` 导入

你有没有见过这种代码？

```go
import (
    "database/sql"
    _ "github.com/go-sql-driver/mysql" // 注意前面的下划线
)
```

这里的 `_` 意思是：**"我引入这个包，我不直接使用它里面的函数，但我希望它执行它的 `init()` 函数。"**

在 MySQL 驱动的源码里，通常会在 `init()` 函数里把自己"注册"给 `database/sql`。如果你不 import 它，`init` 不执行，数据库就连不上。

这就是 `init` 最典型的应用场景（**副作用导入**）。

---

## 6. 多个 init 的执行顺序

### 同一文件多个 init

```go
package main

import "fmt"

func init() {
    fmt.Println("第一个 init")
}

func init() {
    fmt.Println("第二个 init")
}

func main() {
    fmt.Println("main")
}
// 输出顺序：第一个 init -> 第二个 init -> main
```

### 多个包的 init

```
main 包
├── import A 包
│   └── import B 包
│       └── B.init()
│   └── A.init()
└── main.init()
└── main()
```

**执行顺序：** 依赖最深的包最先初始化（B -> A -> main）

---

## 7. 总结

| 函数 | 作用 | 调用方式 |
| --- | --- | --- |
| `main()` | 程序的入口 | 系统自动调用 |
| `init()` | 包的入口（铺路、检查、注册） | 系统自动调用，在 `main` 之前 |

你没学过 `init` 是因为简单的程序不需要它，但稍微复杂一点的项目（尤其是涉及数据库、配置加载、Web 框架）到处都在用它。

---

## 相关阅读

- [Go 基础语法](./basics-syntax.md) - Go 语言基础
- [Go 生态系统](./ecosystem.md) - 常用库和框架
