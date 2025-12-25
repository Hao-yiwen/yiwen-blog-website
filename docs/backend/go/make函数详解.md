---
sidebar_position: 8
title: Make 函数详解
tags: [go, make, slice, map, channel, 内置函数]
---

# Go Make 内置函数详解

在 Go 语言中，`make` 是一个内置函数（Built-in function），它的作用非常专注且独特。

简单来说，**`make` 专门用于初始化 Go 语言中的三种"引用类型"：**

1. **切片 (Slice)**
2. **映射 (Map)**
3. **通道 (Channel)**

**关键区别：** 与 `new` 不同（`new` 只分配内存并返回指针），`make` 不仅分配内存，还会**初始化**这些类型底层的复杂数据结构，并且返回的是**值本身**（而不是指针）。

下面我分别介绍 `make` 在这三种类型中的具体用法和原理。

---

## 1. 创建切片 (Slice)

切片本质上是一个包含三个字段的结构体：指向底层数组的指针、长度 (`len`) 和容量 (`cap`)。`make` 的作用就是创建这个底层数组，并初始化这三个字段。

**语法：**

```go
make([]Type, len, cap) // cap 是可选的，如果省略，cap 等于 len
```

**示例：**

```go
func main() {
    // 1. 创建一个长度为 5，容量为 5 的切片，初始值为 0
    s1 := make([]int, 5)

    // 2. 创建一个长度为 3，但底层容量为 10 的切片
    // 意味着前3个元素可以直接访问，后面7个位置已经分配了内存但暂不可用（需要 append）
    s2 := make([]int, 3, 10)

    fmt.Printf("s2: len=%d, cap=%d\n", len(s2), cap(s2))
    // 输出: s2: len=3, cap=10
}
```

> **为什么需要它？**
> 如果只是声明 `var s []int`，切片的值是 `nil`，没有底层数组，虽然可以直接使用 `append`（因为 `append` 会处理 nil 切片），但如果你想直接通过索引 `s[0] = 1` 赋值，会报错。使用 `make` 可以预先分配好空间。

---

## 2. 创建映射 (Map)

Map 的底层是一个复杂的哈希表。`make` 会初始化哈希表的结构，使其准备好存储键值对。

**语法：**

```go
make(map[KeyType]ValueType, size) // size 是可选的容量提示
```

**示例：**

```go
func main() {
    // 1. 标准初始化
    m1 := make(map[string]int)
    m1["age"] = 30

    // 2. 带容量提示的初始化
    // 如果你预知大概有 100 个元素，提前分配可以减少后续扩容带来的性能开销
    m2 := make(map[string]int, 100)
    m2["id"] = 1
}
```

> **重要警告 (Common Pitfall)：**
> 这一步对于 Map 是**必须**的。
> 如果你只声明 `var m map[string]int`，此时 `m` 是 `nil`。虽然读取 nil map 是安全的（返回零值），但**向 nil map 写入数据会导致程序 Panic（崩溃）**。必须使用 `make` 初始化后才能写入。

---

## 3. 创建通道 (Channel)

Channel 是 Goroutine 之间通信的管道。`make` 用于初始化通道的缓冲区结构。

**语法：**

```go
make(chan Type, bufferSize) // bufferSize 可选
```

**示例：**

```go
func main() {
    // 1. 无缓冲通道 (Unbuffered Channel)
    // 发送和接收必须同步进行，否则会阻塞
    ch1 := make(chan int)

    // 2. 有缓冲通道 (Buffered Channel)
    // 缓冲区大小为 5，发送方在缓冲区满之前不会阻塞
    ch2 := make(chan int, 5)
}
```

---

## 核心面试题：`make` vs `new`

这是 Go 语言中最经典的对比，请看下表：

| 特性 | `make` | `new` |
| --- | --- | --- |
| **适用类型** | 仅限 **slice**, **map**, **channel** | 任何类型 (`int`, `struct`, `array` 等) |
| **返回值** | 返回类型本身 **`T`** (引用) | 返回指向该类型的指针 **`*T`** |
| **主要作用** | 分配内存 + **初始化内部结构** | 仅分配内存 + **置零** (Zero Value) |
| **内存状态** | 准备好可以使用 (Ready to use) | 只是清零的内存，对于 map 等复杂类型可能还需要进一步初始化 |

**简单记忆法：**

* **`make`** 是去工厂**制造**一个复杂的成品（只要是这三样东西，必须用 make）。
* **`new`** 只是去申请一块**新**的空地（给内存置零，返回地址）。

---

## 代码对比示例

```go
func main() {
    // --- 针对指针 (new) ---
    p := new(int)   // p 是 *int 类型，指向一个值为 0 的 int
    fmt.Println(*p) // 输出 0
    *p = 2          // 合法

    // --- 针对 Map (错误示范 vs 正确示范) ---

    // 错误：使用 new 创建 map
    // mPtr 是 *map[string]int，它指向一个 nil 的 map
    mPtr := new(map[string]int)
    // (*mPtr)["key"] = 1 // ❌ Panic! 试图向 nil map 写数据

    // 正确：使用 make 创建 map
    // m 是 map[string]int，它是一个已经初始化好的哈希表
    m := make(map[string]int)
    m["key"] = 1      // ✅ 正确
}
```

---

## 总结

* 如果你要创建 **Slice**、**Map** 或 **Channel**，请毫不犹豫地使用 **`make`**。
* 它不仅分配了内存，还完成了这三种类型正常工作所必须的"幕后"初始化工作。
