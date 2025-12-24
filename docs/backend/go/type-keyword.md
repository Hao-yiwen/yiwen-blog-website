---
sidebar_position: 7
title: Type Keyword
tags: [go, type, 类型定义, 类型别名, 接口, 结构体]
---

# Go Type 关键字详解

在 Go 语言（Golang）中，`type` 关键字非常核心，它不仅用于定义数据的结构，还是 Go 语言实现面向对象编程（OOP）特性的基石。

`type` 的主要用途可以分为以下几大类。为了让你更直观地理解，我将通过代码示例和对比来详细介绍。

---

## 1. 类型定义 (Type Definition)

这是最常见的用法。你可以基于现有的类型（如 `int`, `string`, `struct` 等）创建一个**全新的类型**。

**关键点：** 新类型与原类型在编译期被视为**完全不同**的类型，不能直接隐式赋值，必须进行强制转换。

```go
// 定义一个名为 Celsius 的新类型，底层是 float64
type Celsius float64

func main() {
    var temp Celsius = 25.5
    var num float64 = 10.0

    // num = temp // ❌ 错误：类型不匹配 (cannot use temp (type Celsius) as type float64)

    num = float64(temp) // ✅ 正确：显式转换
    fmt.Println(num)
}
```

> **为什么要这样做？**
> 这样做可以赋予基本类型特定的语义（例如区分"温度"和"普通浮点数"），并且可以为 `Celsius` 类型绑定特有的方法（Method）。

---

## 2. 类型别名 (Type Alias)

这是 Go 1.9 版本引入的特性。它主要用于代码重构或兼容性。

**关键点：** 别名与原类型是**完全一样**的，只是名字不同。它们之间可以直接赋值，不需要转换。

**语法：** 注意中间多了一个 **`=`** 号。

```go
// MyInt 是 int 的别名
type MyInt = int

func main() {
    var a MyInt = 10
    var b int = 20

    b = a // ✅ 正确：因为 MyInt 就是 int，编译器视为同一类型
    fmt.Println(b)
}
```

**内置别名例子：**

* `byte` 是 `uint8` 的别名。
* `rune` 是 `int32` 的别名。

---

## 3. 结构体定义 (Struct Definition)

`type` 最强大的用途之一是定义结构体。这是 Go 语言中聚合数据的主要方式，类似于其他语言中的 Class（类）。

```go
// 定义一个 User 结构体
type User struct {
    Name string `json:"name"` // 还可以添加 Tag
    Age  int
}

// 嵌套结构体（组合）
type Employee struct {
    User      // 嵌入 User，Employee 自动拥有 Name 和 Age
    Title string
}
```

---

## 4. 接口定义 (Interface Definition)

Go 语言的接口是隐式实现的（Duck Typing），`type` 用于定义接口的方法签名集合。

```go
// 定义一个 Reader 接口
type Reader interface {
    Read(p []byte) (n int, err error)
}

// 定义一个空接口（可以代表任何类型）
type Any interface{}
```

---

## 5. 函数类型 (Function Type)

在 Go 中，函数是一等公民。你可以使用 `type` 给某种特定的函数签名定义一个名字。这在编写中间件、回调函数或处理程序时非常有用。

```go
// 定义一个函数类型，接受两个 int，返回一个 int
type MathOp func(int, int) int

// 一个符合 MathOp 签名的具体函数
func add(a, b int) int {
    return a + b
}

func main() {
    var op MathOp = add
    fmt.Println(op(1, 2))
}
```

---

## 6. 为类型定义方法 (Method Receiver)

虽然这不是 `type` 关键字的直接语法，但只有通过 `type` 定义的**命名类型**（Named Type），才能定义与之绑定的方法。

**注意：** 你不能给非本地类型（如直接给 `int` 或其他包的类型）定义方法，必须先定义一个你自己的类型。

```go
type MyList []int // 基于切片定义新类型

// 给 MyList 绑定一个 Sum 方法
func (l MyList) Sum() int {
    sum := 0
    for _, v := range l {
        sum += v
    }
    return sum
}
```

---

## 总结与对比：类型定义 vs 类型别名

这是面试中最高频的考点，请参考下表：

| 特性 | 类型定义 (`type A B`) | 类型别名 (`type A = B`) |
| --- | --- | --- |
| **本质** | 创造了一个**新**的类型 | 只是原类型的一个**新名字** |
| **类型检查** | 严格，不能与原类型混用 | 宽松，与原类型完全互通 |
| **方法继承** | **不继承**原类型的方法 | 拥有原类型的所有方法 |
| **主要用途** | 领域建模、扩展基本类型、定义接口/结构体 | 代码重构、迁移、兼容旧代码 |
