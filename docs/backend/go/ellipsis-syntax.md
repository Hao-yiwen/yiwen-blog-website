---
sidebar_position: 11
title: 省略号(...)语法全解
tags: [go, ellipsis, 可变参数, variadic, append]
---

# Go 语言 `...` (Ellipsis) 用法全解

`...` 在 Go 中主要扮演三种角色：

1. **函数定义时**：把散落的参数**打包**（Pack）成切片。
2. **函数调用时**：把切片**解包**（Unpack）成散落的参数。
3. **数组声明时**：自动计算数组长度。

---

## 1. 场景一：函数定义（可变参数）

**作用：** 接收任意数量的同类型参数。
**位置：** 放在参数类型的**前面**（`...Type`）。
**本质：** 编译器会自动创建一个**切片**来存放这些参数。

### 语法规则

* 只能放在参数列表的**最后**。
* 一个函数只能有一个可变参数。

### 示例代码

```go
package main

import "fmt"

// 1. 基础用法：接收任意个 int
// nums 在函数内部是一个 []int 切片
func Sum(nums ...int) int {
    total := 0
    fmt.Printf("接收到的切片类型: %T, 长度: %d\n", nums, len(nums))

    for _, num := range nums {
        total += num
    }
    return total
}

// 2. 混合用法：固定参数 + 可变参数
// prefix 是必填的，msgs 可以填0个或多个
func Logger(prefix string, msgs ...string) {
    fmt.Printf("[%s] ", prefix)
    for _, msg := range msgs {
        fmt.Print(msg + " ")
    }
    fmt.Println()
}

func main() {
    // 调用 Sum
    fmt.Println("总和:", Sum(1, 2, 3))       // 传3个
    fmt.Println("总和:", Sum(10))            // 传1个
    fmt.Println("总和:", Sum())              // 传0个，nums 是 nil 切片

    // 调用 Logger
    Logger("INFO", "用户登录", "IP:192.168.1.1")
}
```

---

## 2. 场景二：函数调用（参数打散）

**作用：** 将一个现有的切片打散，传递给一个可变参数函数。
**位置：** 放在变量名的**后面**（`slice...`）。
**本质：** 告诉编译器"不要把这个切片当成一个整体参数，把它拆开一个个传进去"。

### 最常见场景：`append` 合并切片

这是 `...` 最常用的地方。由于 `append` 的定义是 `func append(slice []T, elements ...T) []T`，当你试图把一个切片追加到另一个切片时，必须使用 `...`。

### 示例代码

```go
package main

import "fmt"

func main() {
    sliceA := []int{1, 2, 3}
    sliceB := []int{4, 5, 6}

    // ❌ 错误写法
    // append(sliceA, sliceB)
    // 报错：cannot use sliceB (type []int) as type int in append

    // ✅ 正确写法：使用 ... 把 sliceB 里的元素倒出来
    combined := append(sliceA, sliceB...)

    fmt.Println(combined) // 输出 [1 2 3 4 5 6]

    // 另一个例子：传递给 Printf
    args := []interface{}{"Gemini", 18}
    // fmt.Printf("Name: %s, Age: %d", args)    // ❌ 错，args 被当成了一个 struct/slice 打印
    fmt.Printf("Name: %s, Age: %d\n", args...) // ✅ 对，"Gemini" 给 %s, 18 给 %d
}
```

---

## 3. 场景三：数组初始化（自动推断长度）

**作用：** 让编译器根据你写的元素个数，自己数一数数组有多长。
**位置：** 放在数组长度的位置（`[...]Type`）。
**注意：** 产生的是**数组（Array）**，不是切片（Slice）。

### 示例代码

```go
package main

import "fmt"

func main() {
    // 1. 普通数组声明（显式指定长度）
    var arr1 [3]int = [3]int{1, 2, 3}

    // 2. 切片声明（中括号里为空）
    var slice1 []int = []int{1, 2, 3}

    // 3. 使用 ... 自动推断长度的数组
    // 编译器数了一下后面有4个元素，自动把它变成 [4]int 类型
    arr2 := [...]int{10, 20, 30, 40}

    fmt.Printf("arr1 类型: %T (长度写死)\n", arr1)   // [3]int
    fmt.Printf("slice1 类型: %T (切片)\n", slice1)    // []int
    fmt.Printf("arr2 类型: %T (自动推断)\n", arr2)   // [4]int
}
```

---

## 4. 进阶：`...interface{}` (任意类型参数)

这是 Go 语言实现"泛型"参数（在真泛型出来之前）或"任意参数"的标准方式。`fmt.Printf` 和 `log.Println` 都是基于此实现的。

### 示例：实现一个简单的 Println

如果你想包装 `fmt` 包，必须学会透传 `...`。

```go
package main

import "fmt"

// 接收任意数量、任意类型的参数
func MyPrintln(args ...interface{}) {
    // 此时 args 是 []interface{}
    // 必须使用 ... 再次解包传给 fmt，否则 fmt 会打印出中括号 []
    fmt.Println(args...)
}

func main() {
    MyPrintln("Hello", 100, true, 3.14)
    // 输出: Hello 100 true 3.14
}
```

---

## 5. 总结速查表

| 写法 | 场景 | 含义 | 比喻 |
| --- | --- | --- | --- |
| `nums ...int` | 函数定义 (参数表) | **打包**：把散落的参数收集成一个 slice | 吸尘器 (把地上的灰尘吸进肚子里) |
| `slice...` | 函数调用 (传参时) | **解包**：把 slice 里的元素一个个拿出来传进去 | 倒箱子 (把箱子里的球倒出来) |
| `[...]int` | 变量初始化 | **计数**：根据元素个数确定数组长度 | 自动计数器 |

### 核心注意事项

1. **只能在最后：** `func(a int, b ...string)` 是对的，`func(b ...string, a int)` 是错的。
2. **Pass by Value：** 当你使用 `nums ...int` 时，函数内部的 `nums` 是一个切片。虽然切片本身很小（Header），但它指向的底层数组与外部（如果是由 `slice...` 传入的）是共享内存的。小心修改。

---

## 相关阅读

- [切片传参的值拷贝陷阱](./slice-pass-by-value.md) - 理解切片共享底层数组的影响
- [切片内部结构](./slice-internals.md) - 深入理解 SliceHeader
