---
sidebar_position: 7
title: make 函数
tags: [go, make, slice, map, channel, 内置函数]
---

# Go make 函数详解

在 Go 语言中，`make` 是一个内置函数（built-in function），它的作用非常专一且重要：**专门用于初始化切片（Slice）、映射（Map）和通道（Channel）这三种引用类型。**

与简单的变量声明不同，`make` 不仅分配内存，还会对这些复杂的数据结构进行**初始化**（例如初始化哈希表的内部结构、设置切片的底层数组指针等），并返回该类型的**值**（而不是指针）。

## 1. make 的三大应用场景

### A. 创建切片 (Slice)

切片是基于数组的动态封装。使用 `make` 创建切片时，你可以指定**长度 (len)** 和 **容量 (cap)**。

**语法：** `make([]T, length, capacity)`

**参数：**
- `length`: 切片初始包含的元素个数（可以通过索引直接访问）
- `capacity` (可选): 底层数组预分配的空间大小。如果省略，默认等于 `length`

```go
// 1. 指定长度为 5，容量为 10
s1 := make([]int, 5, 10)
fmt.Printf("len=%d, cap=%d, val=%v\n", len(s1), cap(s1), s1)
// 输出: len=5, cap=10, val=[0 0 0 0 0] (前5个元素被初始化为零值)

// 2. 只指定长度（容量默认等于长度）
s2 := make([]int, 5)
// 输出: len=5, cap=5
```

> **为什么要指定容量？**
> 它可以避免切片在 `append` 扩容时频繁重新分配内存和拷贝数据，从而提高性能。

### B. 创建映射 (Map)

Map 是键值对集合。虽然 `var m map[string]int` 可以声明一个 map，但它是 `nil`，直接赋值会报错（panic）。**必须使用 `make` 初始化后才能使用。**

**语法：** `make(map[KeyType]ValueType, capacity)`

**参数：**
- `capacity` (可选): 初始容量提示

```go
// 1. 标准初始化
m1 := make(map[string]int)
m1["age"] = 18 // 正常赋值

// 2. 指定初始容量 (推荐在已知大小时使用)
m2 := make(map[string]int, 100)
```

> **注意：** Map 会根据存入元素的数量自动扩容，但如果你预先知道大概有多少个元素，传入 `capacity` 参数可以减少底层的扩容和哈希重排操作。

### C. 创建通道 (Channel)

Channel 用于 Goroutine 之间的通信。

**语法：** `make(chan Type, buffer_size)`

**参数：**
- `buffer_size` (可选): 缓冲区大小

```go
// 1. 无缓冲通道 (同步通道)
// 发送方发送数据时，必须有接收方在等待，否则发送方阻塞
c1 := make(chan int)

// 2. 有缓冲通道 (异步通道)
// 只要缓冲区未满，发送方就不会阻塞
c2 := make(chan int, 10)
```

## 2. make vs new (高频面试题)

Go 语言中还有一个 `new` 关键字，初学者容易混淆。它们的区别非常关键：

| 特性 | `make` | `new` |
| --- | --- | --- |
| **适用类型** | 仅限 **Slice**, **Map**, **Channel** | 任意类型 (Int, Struct, Array 等) |
| **返回值** | 返回 **引用类型本身** (T) | 返回 **指向零值的指针** (*T) |
| **主要作用** | 分配内存 + **初始化内部结构** (非零值) | 仅分配内存 + **置零** (Zero Value) |
| **内存填充** | 填充该类型的初始化状态 | 填充为 0 (0, false, nil 等) |

**代码对比：**

```go
// --- 使用 new ---
// new 返回指针，且内存置零
p := new([]int)
fmt.Println(p) // 输出: &[] (一个指向 nil 切片的指针)
// *p = append(*p, 1) // 这样用虽然可以，但很别扭，通常很少对 slice 用 new

// --- 使用 make ---
// make 返回初始化好的值，可以直接用
v := make([]int, 0)
v = append(v, 1) // 正常使用
```

## 3. 底层原理简述

为什么 Slice、Map、Channel 必须用 `make`？

因为这三种类型不仅仅是一块连续的内存，它们是**复杂的结构体**：

- **Slice:** 包含一个指向底层数组的指针、长度和容量。如果只申请内存不初始化，指针就是 nil，无法存数据。
- **Map:** 内部需要初始化哈希桶（buckets）、哈希种子等。
- **Channel:** 内部包含发送队列、接收队列、互斥锁（Mutex）等。

`make` 的工作就是把这些"零件"组装好，让你可以直接安全地使用它们。

## 4. 总结

1. **只用于三个类型：** `slice`, `map`, `chan`
2. **不返回指针：** 返回的是可以直接使用的引用值
3. **防止 Panic：** 尤其是 Map，只声明不 `make` 直接赋值会导致 panic
