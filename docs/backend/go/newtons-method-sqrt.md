---
sidebar_position: 5
title: 牛顿法求平方根
tags: [go, 算法, 牛顿法, 数学, 循环]
---

# 牛顿法求平方根

牛顿法（Newton's Method）是一种快速逼近函数零点的迭代算法。在求平方根时，它比暴力搜索快得多。

## 数学原理

要求 $\sqrt{x}$，即求方程 $z^2 - x = 0$ 的解。

牛顿法迭代公式：

$$
z_{n+1} = z_n - \frac{f(z_n)}{f'(z_n)}
$$

对于 $f(z) = z^2 - x$，有 $f'(z) = 2z$，代入得：

$$
z_{n+1} = z_n - \frac{z_n^2 - x}{2z_n}
$$

## Go 实现

```go
package main

import (
    "fmt"
    "math"
)

func Sqrt(x float64) float64 {
    z := 1.0
    for math.Abs(z*z-x) > 1e-10 {  // 误差大于阈值时继续
        z -= (z*z - x) / (2 * z)   // 牛顿法公式
    }
    return z
}

func main() {
    fmt.Println(Sqrt(2))       // 1.4142135623730951
    fmt.Println(math.Sqrt(2))  // 1.4142135623730951
}
```

## 关键要点

### 1. 循环条件用 `>` 不是 `<`

```go
// ✅ 正确：误差大于阈值时继续迭代
for math.Abs(z*z-x) > 1e-10 {

// ❌ 错误：这会立即停止
for math.Abs(z*z-x) < 1e-10 {
```

### 2. 必须取绝对值

```go
// ✅ 正确：取绝对值
for math.Abs(z*z-x) > 1e-10 {

// ❌ 错误：z² - x 可能是负数
for z*z-x > 1e-10 {
```

为什么？当 z 逼近正确值时：
- 如果 z 偏大：$z^2 - x > 0$
- 如果 z 偏小：$z^2 - x < 0$

不取绝对值的话，当 z 偏小时条件永远为 false，循环会提前退出。

## 逐步迭代过程

以 $\sqrt{2}$ 为例：

| 迭代次数 | z 值 | z² | 误差 |
|---------|------|-----|------|
| 初始 | 1.0 | 1.0 | 1.0 |
| 1 | 1.5 | 2.25 | 0.25 |
| 2 | 1.4166... | 2.0069... | 0.0069... |
| 3 | 1.4142... | 2.0000... | 0.000006... |
| 4 | 1.4142135... | ≈ 2 | < 1e-10 |

只需约 4 次迭代就能达到很高精度！

## 变体实现

### 固定迭代次数版本

```go
func Sqrt(x float64) float64 {
    z := 1.0
    for i := 0; i < 10; i++ {
        z -= (z*z - x) / (2 * z)
    }
    return z
}
```

### 更好的初始值

```go
func Sqrt(x float64) float64 {
    z := x / 2  // 初始值设为 x/2，对大数收敛更快
    for math.Abs(z*z-x) > 1e-10 {
        z -= (z*z - x) / (2 * z)
    }
    return z
}
```

## 与标准库对比

```go
func main() {
    testCases := []float64{2, 9, 100, 0.25, 1e10}

    for _, x := range testCases {
        ours := Sqrt(x)
        std := math.Sqrt(x)
        diff := math.Abs(ours - std)
        fmt.Printf("√%.2f: 我们=%.10f, 标准库=%.10f, 差值=%.2e\n",
            x, ours, std, diff)
    }
}
```

输出：

```
√2.00: 我们=1.4142135624, 标准库=1.4142135624, 差值=2.98e-13
√9.00: 我们=3.0000000000, 标准库=3.0000000000, 差值=0.00e+00
√100.00: 我们=10.0000000000, 标准库=10.0000000000, 差值=0.00e+00
√0.25: 我们=0.5000000000, 标准库=0.5000000000, 差值=0.00e+00
```

## 边界情况处理

```go
func Sqrt(x float64) float64 {
    if x < 0 {
        return math.NaN()  // 负数返回 NaN
    }
    if x == 0 {
        return 0
    }

    z := x / 2
    if z == 0 {
        z = 1  // 防止 x 太小导致 z 为 0
    }

    for math.Abs(z*z-x) > 1e-10*x {  // 相对误差
        z -= (z*z - x) / (2 * z)
    }
    return z
}
```

## 总结

| 要点 | 说明 |
|------|------|
| 核心公式 | $z = z - \frac{z^2 - x}{2z}$ |
| 循环条件 | 误差**大于**阈值时继续（用 `>`） |
| 绝对值 | 必须用 `math.Abs()`，因为误差可正可负 |
| 收敛速度 | 二次收敛，约 4 次迭代达到高精度 |
| 初始值 | `z = 1.0` 或 `z = x/2` 都可以 |

## 参考

- [A Tour of Go - Exercise: Loops and Functions](https://go.dev/tour/flowcontrol/8)
- [Newton's method - Wikipedia](https://en.wikipedia.org/wiki/Newton%27s_method)
