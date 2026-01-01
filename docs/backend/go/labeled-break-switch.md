---
sidebar_position: 5
title: 带标签的 Break 与 Switch
tags: [go, syntax, control-flow, switch]
---

# Go 带标签的 Break 与复杂流控制

本文通过一个典型的**变长编码解码器**代码片段，展示 Go 语言中处理复杂流控制的三个核心技巧：

1. **带标签的 Break (`break Loop`)**
2. **不带表达式的 Switch**
3. **步长动态变化的 For 循环**

## 示例代码

```go
Loop:
    for n := 0; n < len(src); n += size {
        switch {
        case src[n] < sizeOne:
            if validateOnly {
                break
            }
            size = 1
            update(src[n])

        case src[n] < sizeTwo:
            if n+1 >= len(src) {
                err = errShortInput
                break Loop
            }
            if validateOnly {
                break
            }
            size = 2
            update(src[n] + src[n+1]<<shift)
        }
    }
```

这种代码通常出现在处理网络协议、压缩算法（如 Huffman 解码）或序列化格式（如 Protobuf、Varint）的场景中。

## 1. 动态步长的 For 循环

```go
for n := 0; n < len(src); n += size {
```

- **含义**：步进不是固定的 1，而是 `n += size`
- **目的**：处理变长数据。有的数据占 1 字节，有的占 2 字节（甚至更多）。程序先读第一个字节判断数据长度（`size`），处理完后跳过这么多字节，读取下一个数据

## 2. 带标签的 Break

### 什么是标签（Label）？

```go
Loop:
    for ... {
        switch {
            case ...:
                break Loop // 跳出标记为 Loop 的循环
        }
    }
```

`Loop:` 是一个**标签（Label）**，用于标识某个循环或语句块。

### 为什么需要它？

在 Go 中，`break` 默认只会跳出**最近的一层** `switch`、`select` 或 `for`：

```go
for {
    switch x {
    case 1:
        break  // 只跳出 switch，for 循环继续
    }
    // 代码会执行到这里
}
```

但如果遇到严重错误需要**直接终止整个 For 循环**，就必须用 `break Loop`：

```go
Loop:
    for {
        switch x {
        case 1:
            break Loop  // 跳出整个 for 循环
        }
        // 这里不会被执行
    }
// 代码跳转到这里
```

### 对比示例

```go
// ❌ 普通 break - 只跳出 switch
for i := 0; i < 10; i++ {
    switch i {
    case 5:
        break  // 跳出 switch，继续下一次 for 循环
    }
    fmt.Println(i)  // 0,1,2,3,4,5,6,7,8,9 都会打印
}

// ✅ 带标签的 break - 跳出整个 for
Loop:
    for i := 0; i < 10; i++ {
        switch i {
        case 5:
            break Loop  // 跳出整个 for 循环
        }
        fmt.Println(i)  // 只打印 0,1,2,3,4
    }
```

## 3. 不带表达式的 Switch

Go 的 `switch` 可以不带表达式，等价于 `switch true`：

```go
switch {
case src[n] < sizeOne:
    // 处理小数据
case src[n] < sizeTwo:
    // 处理大数据
default:
    // 其他情况
}
```

这种写法相当于一系列的 `if-else if-else`，但更清晰：

```go
// 等价于
if src[n] < sizeOne {
    // ...
} else if src[n] < sizeTwo {
    // ...
} else {
    // ...
}
```

## 4. 代码逻辑拆解

### Case 1: 数据较小，只占 1 字节

```go
case src[n] < sizeOne:
    if validateOnly {
        break  // 校验模式，跳出 switch，进入下一次循环
    }
    size = 1
    update(src[n])
```

### Case 2: 数据较大，占 2 字节

```go
case src[n] < sizeTwo:
    if n+1 >= len(src) {
        err = errShortInput
        break Loop  // 数据不够，直接终止整个循环
    }
    if validateOnly {
        break
    }
    size = 2
    update(src[n] + src[n+1]<<shift)
```

## 5. 标签的其他用法

### continue 也可以带标签

```go
OuterLoop:
    for i := 0; i < 3; i++ {
        for j := 0; j < 3; j++ {
            if j == 1 {
                continue OuterLoop  // 跳过外层循环的当前迭代
            }
            fmt.Println(i, j)
        }
    }
// 输出: 0 0, 1 0, 2 0
```

### goto 语句

Go 也支持 `goto`，但不推荐使用：

```go
func example() {
    goto End
    fmt.Println("这行不会执行")
End:
    fmt.Println("跳转到这里")
}
```

## 总结

这段代码展示了三个重要技巧：

| 技巧 | 用途 |
|------|------|
| `n += size` | 动态步长，处理变长数据 |
| `break Loop` | 从深层嵌套中直接跳出指定循环 |
| `switch { }` | 替代复杂的 if-else 链，代码更清晰 |

带标签的 `break` 和 `continue` 是处理多层嵌套循环的优雅方案，避免了使用额外的布尔变量或复杂的条件判断。
