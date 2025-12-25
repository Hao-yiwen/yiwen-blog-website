---
sidebar_position: 4
title: Range 迭代
tags: [go, range, 迭代, for循环]
---

# Go Range 迭代详解

`range` 是 Go 的**迭代关键字**，用于遍历各种数据结构。它会根据数据类型返回不同的东西。

## 不同类型的 range

| 数据类型 | range 返回 | 示例 |
|----------|-----------|------|
| **数组/切片** | 索引, 值 | `for i, v := range []int{1,2,3}` |
| **字符串** | 字节索引, rune | `for i, v := range "你好"` |
| **map** | 键, 值 | `for k, v := range map[string]int{}` |
| **channel** | 值 | `for v := range ch` |

## 代码示例

### 1. 切片

```go
nums := []int{10, 20, 30}
for i, v := range nums {
    // i = 0, 1, 2（索引）
    // v = 10, 20, 30（值）
}
```

### 2. 字符串

```go
for i, v := range "Hi你" {
    // i = 0, v = 'H'
    // i = 1, v = 'i'
    // i = 2, v = '你'（注意：下一个索引会是 5，因为"你"占3字节）
}
```

### 3. map

```go
m := map[string]int{"a": 1, "b": 2}
for k, v := range m {
    // k = "a", v = 1
    // k = "b", v = 2
}
```

### 4. 只要索引/键（忽略值）

```go
for i := range nums { }
```

### 5. 只要值（忽略索引）

```go
for _, v := range nums { }
```

## 字符串的特殊之处

```go
s := "你好"

// range 遍历：按 rune（字符）遍历
for i, v := range s {
    // i 是字节位置：0, 3
    // v 是完整字符：'你', '好'
}

// 普通 for 遍历：按字节遍历
for i := 0; i < len(s); i++ {
    // s[i] 是单个字节，中文会乱码
}
```

## 总结

`range` = "遍历这个东西，给我每一项的（位置, 值）"

- 对于数组/切片：位置是索引，值是元素
- 对于字符串：位置是字节索引，值是 rune（完整字符）
- 对于 map：位置是键，值是对应的值
- 对于 channel：只有值，没有位置
