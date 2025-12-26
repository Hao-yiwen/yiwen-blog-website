---
sidebar_position: 6
title: nil切片与空切片
tags: [go, slice, nil, make, 内存]
---

# Nil 切片 vs 空切片

**一句话总结：`var` 只是声明了一个"空壳"，而 `make` 则是真正"分配了内存"并填充了这个壳。**

## 切片的底层结构

Go 语言中的切片（Slice）在运行时是一个结构体（Slice Header），包含三个字段：

```go
type SliceHeader struct {
    Data uintptr  // 指向底层数组的内存地址
    Len  int      // 当前包含多少个元素
    Cap  int      // 底层数组能装多少个元素
}
```

## 两种创建方式对比

### A. `var s []int`（直接声明）

声明变量但**没有初始化**，Go 会给它赋予"零值"：

| 字段 | 值 |
|------|-----|
| Data | `nil` (0x0) |
| Len | 0 |
| Cap | 0 |

**结果**：整个切片等于 `nil`，像一个没有任何内容的空指针。

### B. `make([]int, 0)`（使用 make）

`make` 会**初始化**切片，哪怕长度是 0：

| 字段 | 值 |
|------|-----|
| Data | **非 nil**（指向 `zerobase`） |
| Len | 0 |
| Cap | 0 |

**结果**：切片不等于 `nil`，因为指针字段指向了真实存在的内存地址。

## 形象比喻

| 方式 | 比喻 |
|------|------|
| `var s []int` (nil) | 你只是脑子里想去吃饭，但连餐厅门都没进。你**没有**桌子。 |
| `make([]int, 0)` (空切片) | 你进了餐厅，服务员给你**分配了一张桌子**。虽然桌子上现在是空的，但桌子实实在在存在。 |

## 代码验证：看内存地址

```go
package main

import (
    "fmt"
    "reflect"
    "unsafe"
)

func main() {
    // 1. var 声明 (Nil 切片)
    var nilSlice []int
    nilHeader := (*reflect.SliceHeader)(unsafe.Pointer(&nilSlice))

    // 2. make 创建 (Empty 切片)
    makeSlice := make([]int, 0)
    makeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&makeSlice))

    fmt.Println("--- 比较 ---")
    fmt.Printf("Nil Slice 指针地址: 0x%x (即 nil)\n", nilHeader.Data)
    fmt.Printf("Make Slice 指针地址: 0x%x (非 nil)\n", makeHeader.Data)

    fmt.Println("\n--- 逻辑判断 ---")
    fmt.Println("nilSlice == nil?", nilSlice == nil)   // true
    fmt.Println("makeSlice == nil?", makeSlice == nil) // false
}
```

**输出：**

```
--- 比较 ---
Nil Slice 指针地址: 0x0           <-- 确实是 nil
Make Slice 指针地址: 0xc000067f50 <-- 确实分配了地址

--- 逻辑判断 ---
nilSlice == nil? true
makeSlice == nil? false
```

## 字面量 `[]int{}`

除了 `make`，字面量创建也是非 nil：

```go
s := []int{} // 直接字面量创建
fmt.Println(s == nil) // false
```

这和 `make([]int, 0)` 几乎一样，都会分配一个非 nil 的指针。

## 总结对照表

| 创建方式 | 代码 | 指针 (Data) | 是否为 nil | 含义 |
|----------|------|-------------|-----------|------|
| **声明 (Var)** | `var s []int` | **0 (nil)** | **Yes** | 还没初始化，什么都没有 |
| **Make** | `make([]int, 0)` | **非 0** | **No** | 初始化了，只是没装东西 |
| **字面量** | `[]int{}` | **非 0** | **No** | 初始化了，只是没装东西 |

## 功能上的区别

虽然内存结构不同，但在**大多数操作上表现一致**：

```go
var nilSlice []int
makeSlice := make([]int, 0)

// ✅ 都可以 append
nilSlice = append(nilSlice, 1)
makeSlice = append(makeSlice, 1)

// ✅ 都可以获取长度
len(nilSlice)  // 0
len(makeSlice) // 0

// ✅ 都可以 range（不执行循环体）
for _, v := range nilSlice { }
for _, v := range makeSlice { }
```

## JSON 序列化的区别

这是最常见的实际差异：

```go
import "encoding/json"

var nilSlice []int
makeSlice := make([]int, 0)

nilJSON, _ := json.Marshal(nilSlice)
makeJSON, _ := json.Marshal(makeSlice)

fmt.Println(string(nilJSON))  // null
fmt.Println(string(makeJSON)) // []
```

| 切片类型 | JSON 输出 |
|---------|----------|
| nil 切片 | `null` |
| 空切片 | `[]` |

**API 设计时要注意**：如果 API 需要返回空数组 `[]` 而不是 `null`，必须使用 `make` 或字面量。

## 什么时候用哪个？

### 用 `var`（nil 切片）

```go
var s []int
s = append(s, 1, 2, 3)
```

- 只是声明，以后再用 `append` 添加数据
- 解析 JSON 时（JSON `null` 对应 nil 切片）
- **最推荐**：不需要分配内存，效率最高

### 用 `make`

```go
// 预分配容量
s := make([]int, 0, 100)

// 需要直接索引赋值
s := make([]int, 10)
s[5] = 1
```

- 需要**预先分配容量**提升性能
- 需要直接通过索引赋值
- API 返回需要 `[]` 而非 `null`

## Map 的致命陷阱

切片对 nil 很宽容，但 **Map 不是**：

```go
// 切片：nil 可以 append
var s []int
s = append(s, 1) // ✅ 正常工作

// Map：nil 写入会 panic！
var m map[string]int
m["key"] = 1 // ❌ panic: assignment to entry in nil map
```

**Map 必须初始化后才能写入：**

```go
m := make(map[string]int)
m["key"] = 1 // ✅ 正常工作
```

## 总结

| 要点 | 说明 |
|------|------|
| `var s []int` | 指针为 nil，是真正的"空" |
| `make([]int, 0)` | 指针非 nil，是"初始化过的空" |
| 功能差异 | 大多数操作相同，JSON 序列化不同 |
| 性能 | `var` 不分配内存，更高效 |
| 推荐 | 一般用 `var`，需要预分配或 JSON `[]` 时用 `make` |
