---
sidebar_position: 3
title: 引用类型 vs 值类型速查表
tags: [go, types, 引用类型, 值类型, map, slice, channel, 面试题]
---

# Go 引用类型 vs 值类型速查表

Go 中传递变量时，有些类型会复制底层数据（值类型），有些只复制"指针/引用"（引用语义类型）。搞清楚这一点，能避免很多坑。

## 引用语义类型

以下类型传递时**不会复制底层数据**，修改会影响原始数据：

| 类型 | 底层实现 | 说明 |
|------|----------|------|
| **map** | 指向哈希表的指针 | 直接传递即可，无需 `&` |
| **slice** | `{Ptr, Len, Cap}` 结构体 | 指针共享数组，但 Len/Cap 独立 |
| **channel** | 指向通道结构的指针 | 直接传递即可 |
| **pointer** | 指针本身 | `*T` 类型 |
| **function** | 函数引用 | 函数值是引用 |
| **interface** | `{类型信息, 数据指针}` | 内部包含指针 |

## 值类型

以下类型传递时**会完整复制**，修改不影响原始数据：

| 类型 | 说明 |
|------|------|
| `int`, `float64`, `bool` | 基本数字和布尔类型 |
| `string` | 不可变，传递时复制头部结构 |
| `array` | 固定长度数组，**注意不是 slice** |
| `struct` | 结构体会完整复制所有字段 |

## 常见陷阱示例

### 1. slice：共享数组但 Len 独立

```go
func modify(s []int) {
    s[0] = 999      // ✅ 外部能看到，因为共享底层数组
    s = append(s, 4) // ❌ 外部看不到，因为 Len 是独立的副本
}

func main() {
    s := []int{1, 2, 3}
    modify(s)
    fmt.Println(s) // [999 2 3]，长度还是 3
}
```

### 2. array：完全复制

```go
func modify(a [3]int) {
    a[0] = 999 // 修改的是副本
}

func main() {
    a := [3]int{1, 2, 3}
    modify(a)
    fmt.Println(a) // [1 2 3]，不变！
}
```

### 3. struct：完全复制

```go
type Person struct {
    Name string
}

func modify(p Person) {
    p.Name = "Jerry" // 修改的是副本
}

func main() {
    p := Person{Name: "Tom"}
    modify(p)
    fmt.Println(p.Name) // "Tom"，不变！
}
```

### 4. map：直接传就行

```go
func modify(m map[string]int) {
    m["new"] = 100 // ✅ 外部能看到
}

func main() {
    m := map[string]int{"a": 1}
    modify(m)
    fmt.Println(m) // map[a:1 new:100]
}
```

## 什么时候需要传指针？

| 场景 | 是否需要指针 |
|------|--------------|
| 传 `map` | ❌ 不需要，直接传 |
| 传 `slice` 但要 `append` | ✅ 需要 `*[]T` 或返回新 slice |
| 传 `slice` 只修改元素 | ❌ 不需要 |
| 传 `struct` 要修改 | ✅ 需要 `*StructName` |
| 传 `array` 要修改 | ✅ 需要 `*[N]T` |
| 传 `channel` | ❌ 不需要，直接传 |

## 一句话总结

> **map、channel、function 直接传；slice 改元素直接传，append 要返回或传指针；struct、array 要修改必须传指针。**

## 相关阅读

- [基础类型](./basic-types.md) - Go 的基本数据类型
- [切片传参的值拷贝陷阱](./slice-pass-by-value.md) - 深入理解 slice 传参
- [切片内部结构](./slice-internals.md) - SliceHeader 详解
