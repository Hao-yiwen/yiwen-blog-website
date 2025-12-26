---
sidebar_position: 10
title: 字符编码与rune
tags: [go, string, rune, utf-8, unicode, 字符编码]
---

# Go 字符编码：String vs Rune

这是一个非常精准的问题！你触碰到了 **字符编码（Encoding）** 和 **字符集（Character Set）** 的核心区别。

简短的回答是：**你说的都对，但是场景不同。**

- **存储/传输时（String）**：是用 3 个 `byte` (`uint8`) 来表示一个中文字符（在 UTF-8 编码下）。
- **处理/计算时（Rune）**：是用 1 个 `int32` 来表示一个中文字符（这就是 Go 里的 `rune`）。

我们可以把这想象成 **"打包"** 和 **"解包"** 的关系。

## 1. 存储形态：String (UTF-8, `uint8`)

**是为了省空间。**

如果所有的字符（包括简单的英文 'a', 'b'）都用 `int32`（4个字节）存，那原本 1GB 的英文文件就会变成 4GB，太浪费了。

Go 的 `string` 默认使用 **UTF-8** 编码，这是一种 **变长** 编码：

- **英文**：只用 **1个** `byte` (`uint8`)。
- **中文**：通常用 **3个** `byte` (`uint8`)。

所以当你定义 `s := "中"` 时，内存里躺着的是 3 个 `uint8`：
`[11100100, 10111000, 10101101]` (十六进制：E4 B8 AD)

## 2. 逻辑形态：Rune (Unicode, `int32`)

**是为了方便计算。**

当我们想把 "中" 当作 **一个字** 来看待时（比如算长度、截取），我们需要一个统一的编号。这个编号就是 **Unicode 码点（Code Point）**。

在 Go 中，为了装得下全世界所有的字符（几十万个），专门用了一个类型叫 `rune`。
**`rune` 本质上就是 `int32`。**

- "中" 的 Unicode 编号是 `20013`（十六进制 4E2D）。
- 这正好能塞进一个 `int32` 里。

## 3. 代码实锤验证

我们可以写段代码，亲眼看看这"两幅面孔"：

```go
package main

import (
	"fmt"
	"unicode/utf8"
)

func main() {
	s := "中"

	// --- 视角 1: String (底层是 byte/uint8) ---
	// 这里可以看到它占了 3 个字节
	fmt.Printf("字符串长度 (len): %d\n", len(s))
	// 打印底层字节：[228 184 173] (对应十六进制 E4 B8 AD)
	fmt.Printf("底层字节: %v\n", []byte(s))

	fmt.Println("----------------")

	// --- 视角 2: Rune (本质是 int32) ---
	// 把它转换成 rune 切片，相当于"解压缩"了
	r := []rune(s)
	// 这里长度就是 1 了
	fmt.Printf("字符数量 (rune): %d\n", len(r))
	// 打印数值：[20013]
	fmt.Printf("Unicode 数值 (int32): %v\n", r)

	// 也可以用标准库函数直接数
	fmt.Printf("标准库数出来的字数: %d\n", utf8.RuneCountInString(s))
}
```

## 4. 总结对比

| 场景 | Go 类型 | 底层类型 | "中" 的样子 | 为什么？ |
| --- | --- | --- | --- | --- |
| **存储/网络传输** | `string` | `[]uint8` (byte) | `[228, 184, 173]` | **省空间** (UTF-8 变长编码) |
| **内存处理/计数** | `[]rune` | `[]int32` | `[20013]` | **定长方便** (每个字都是独立的 4 字节) |

## 5. Range 遍历的秘密

这就是为什么在 Go 里，如果你用 `range` 遍历字符串，得到的 `value` 会自动变成 `rune` (`int32`)，因为 Go 知道你大概率是想处理"字符"，而不是处理"碎片字节"。

```go
for i, v := range "中" {
    // i 是 0 (字节索引)
    // v 是 20013 (rune/int32，自动帮你转好了)
    fmt.Printf("%d type: %T\n", v, v)
}
```

## 相关阅读

- [字符串设计哲学](./string-design.md) - Go vs Java 的字符串设计差异
- [Range 迭代](./range.md) - Go 的 range 关键字详解
- [基础类型](./basic-types.md) - Go 语言基础类型概览
