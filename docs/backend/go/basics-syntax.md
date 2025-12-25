---
sidebar_position: 3
title: 基础语法详解
tags: [go, syntax, basics]
---

# Go 语言基础语法详解

Go 语言（Golang）以其简洁、高效和并发友好著称。它的语法设计摒弃了许多传统语言的繁琐特性（如分号结尾、复杂的类继承等）。

以下是 Go 语言常用基础语法的详细梳理，包含代码示例和关键点说明。

## 1. 程序结构 (Program Structure)

Go 程序由包（Package）组成。入口文件必须属于 `main` 包，且包含 `main` 函数。

```go
package main // 1. 定义包名

import "fmt" // 2. 导入标准库

// 3. 入口函数
func main() {
    fmt.Println("Hello, World!") // 语句结尾不需要分号
}
```

## 2. 变量与常量 (Variables & Constants)

Go 是强类型语言，但支持类型推导。

### 变量声明

- **标准声明：** `var 变量名 类型`
- **简短声明（常用）：** `变量名 := 值` （只能在函数内部使用）

```go
// 标准声明
var a int = 10
var b = 20 // 自动推导类型

// 批量声明
var (
    name string = "Gemini"
    age  int    = 1
)

func main() {
    // 简短声明 (最常用)
    message := "Go is concise"
    fmt.Println(message)
}
```

### 常量

使用 `const` 关键字，支持 `iota` 枚举器。

```go
const Pi = 3.14159
const (
    StatusOk = 200
    NotFound = 404
)

// iota 自增枚举
const (
    Sunday = iota // 0
    Monday        // 1
    Tuesday       // 2
)
```

## 3. 基本数据类型 (Basic Data Types)

Go 的零值（Zero Value）机制非常重要：变量声明后未赋值，会默认为零值（0, false, "", nil）。

| 类型     | 示例                             | 零值        |
| -------- | -------------------------------- | ----------- |
| **布尔** | `bool`                           | `false`     |
| **字符串** | `string`                       | `""` (空串) |
| **整数** | `int`, `int64`, `uint`, `byte`   | `0`         |
| **浮点** | `float32`, `float64`             | `0.0`       |

## 4. 流程控制 (Control Structures)

### 条件判断 (if)

`if` 的条件不需要括号，大括号 `{}` 是必须的。可以在 `if` 中进行简短声明。

```go
if x := 10; x > 5 {
    fmt.Println("x is big")
} else {
    fmt.Println("x is small")
}
```

### 循环 (for)

Go 只有 `for` 一种循环关键字（没有 `while`）。

```go
// 1. 标准写法
for i := 0; i < 5; i++ {
    fmt.Println(i)
}

// 2. 类似 while 的写法
sum := 1
for sum < 100 {
    sum += sum
}

// 3. 死循环
for {
    // 需配合 break 跳出
}

// 4. Range 遍历 (数组、切片、Map)
nums := []int{2, 3, 4}
for index, value := range nums {
    fmt.Printf("Index: %d, Value: %d\n", index, value)
}
```

### 选择结构 (switch)

默认自带 `break`，除非使用 `fallthrough`。

```go
switch os := "darwin"; os {
case "darwin":
    fmt.Println("OS X.")
case "linux":
    fmt.Println("Linux.")
default:
    fmt.Printf("%s.", os)
}
```

## 5. 组合数据类型 (Collections)

这是 Go 开发中最常用的部分，尤其是切片（Slice）和映射（Map）。

### 数组 (Array) & 切片 (Slice)

- **数组：** 固定长度，值类型。
- **切片：** 动态长度，引用类型（是对数组的封装）。

```go
// 数组 (很少直接用)
var arr [5]int

// 切片 (常用)
slice1 := []int{1, 2, 3}      // 直接初始化
slice2 := make([]int, 5, 10)  // make(类型, 长度, 容量)

// 追加元素
slice1 = append(slice1, 4, 5)

// 切片操作 (左闭右开)
subSlice := slice1[1:3] // 获取索引 1 到 2 的元素
```

### 映射 (Map)

键值对集合，无序。

```go
// 创建 Map
m := make(map[string]int)
m["age"] = 30

// 声明并初始化
scores := map[string]int{
    "Alice": 90,
    "Bob":   85,
}

// 检查键是否存在
if val, ok := scores["Alice"]; ok {
    fmt.Println("Alice's score:", val)
}

// 删除
delete(scores, "Bob")
```

## 6. 函数 (Functions)

Go 函数的一大特色是**支持多返回值**。

```go
// 接收两个 int，返回一个 int
func add(x int, y int) int {
    return x + y
}

// 多返回值 (通常用于返回 结果 + 错误)
func div(x, y int) (int, error) {
    if y == 0 {
        return 0, fmt.Errorf("cannot divide by zero")
    }
    return x / y, nil
}

// 使用
result, err := div(10, 2)
```

## 7. 指针 (Pointers)

Go 保留了指针，但不支持指针运算。主要用于**修改函数外部变量**或**避免大结构体的拷贝**。

- `&`：取地址
- `*`：取值（解引用）

```go
func changeValue(val *int) {
    *val = 100 // 修改指针指向内存的值
}

func main() {
    x := 10
    changeValue(&x) // 传入地址
    fmt.Println(x)  // 输出 100
}
```

## 8. 结构体与方法 (Structs & Methods)

Go 没有 `class`，使用 `struct` 实现面向对象特性。

### 结构体 (Struct)

```go
type Person struct {
    Name string
    Age  int
}

// 初始化
p := Person{Name: "Bob", Age: 20}
```

### 方法 (Method)

方法是绑定到特定类型（接收者 Receiver）的函数。

```go
// (p Person) 是值接收者，(p *Person) 是指针接收者
// 如果需要修改结构体内部的值，必须用指针接收者
func (p *Person) SayHello() {
    fmt.Printf("Hi, I am %s\n", p.Name)
    p.Age++ // 这里的修改会生效
}

func main() {
    p := Person{Name: "Alice", Age: 25}
    p.SayHello()
}
```

## 9. 接口 (Interfaces)

Go 的接口是**隐式实现**的（Duck Typing）：只要一个类型实现了接口定义的所有方法，它就实现了该接口，无需 `implements` 关键字。

```go
// 定义接口
type Speaker interface {
    Speak()
}

// Dog 结构体
type Dog struct{}

// Dog 实现 Speak 方法
func (d Dog) Speak() {
    fmt.Println("Woof!")
}

func main() {
    var s Speaker
    s = Dog{} // Dog 实现了 Speaker，可以直接赋值
    s.Speak()
}
```

:::tip
`interface{}` (空接口) 可以代表任何类型，类似于 Java 的 `Object`。
:::

## 10. 错误处理 (Error Handling)

Go 没有 `try-catch`，而是通过返回值处理错误。同时使用 `defer` 来处理资源释放。

```go
import (
    "os"
    "fmt"
)

func readFile(filename string) error {
    f, err := os.Open(filename)
    if err != nil {
        return err // 遇到错误直接返回
    }
    // defer 会在函数返回前执行，常用于关闭文件/连接
    defer f.Close()

    // ... 读取文件逻辑
    return nil
}
```

## 11. 并发 (Concurrency)

这是 Go 的杀手级特性。

- **Goroutine:** 轻量级线程，使用 `go` 关键字启动。
- **Channel:** 用于 Goroutine 之间的通信。

```go
func worker(c chan string) {
    // 做一些耗时操作
    c <- "Job Done" // 发送数据到通道
}

func main() {
    // 创建一个通道
    msgChan := make(chan string)

    // 启动 Goroutine
    go worker(msgChan)

    // 阻塞等待通道接收数据
    msg := <-msgChan
    fmt.Println(msg)
}
```

## 总结

Go 语言的核心语法特点：

1. **简洁性** - 没有分号、没有括号、没有繁琐的类继承
2. **强类型** - 但支持类型推导，兼顾安全和便利
3. **零值机制** - 所有变量都有明确的默认值
4. **错误处理** - 通过多返回值显式处理，而非异常
5. **并发原语** - Goroutine 和 Channel 是一等公民

掌握以上基础语法，就可以开始编写 Go 程序了。建议配合 [Go Tour](https://go.dev/tour/) 进行实践练习。
