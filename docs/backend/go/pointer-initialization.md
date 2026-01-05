---
sidebar_position: 9
title: new关键字与指针初始化
tags: [go, new, pointer, 指针, 内存分配]
---

# Go new 关键字与指针初始化详解

既然你用了 `new`，那么你手里拿到的是一个**指针**（`*T`），指向的是一块"零值"内存。

要初始化（赋值），你必须通过**解引用（dereference）或者直接字段访问**来把具体的数据填进去。

根据类型的不同，操作方式分为三种情况：

---

## 1. 结构体 (Struct) - 最常用的情况

对于结构体，Go 提供了语法糖，你可以直接用 `.` 来访问字段，编译器会自动帮你处理指针解引用。

```go
type User struct {
    Name string
    Age  int
}

func main() {
    // 1. 使用 new，得到指针 u (类型是 *User)
    u := new(User)

    // 此时 u.Name 是 "", u.Age 是 0

    // 2. 初始化/赋值
    u.Name = "Gemini" // 等同于 (*u).Name = "Gemini"
    u.Age = 18

    fmt.Println(u) // 输出 &{Gemini 18}
}
```

---

## 2. 基本类型 (int, string, bool)

对于基本类型，你必须显式地使用 `*` 符号来给指针指向的内存赋值。

```go
func main() {
    // 1. 使用 new，得到指针 p (类型是 *int)
    p := new(int)

    // 此时 *p 是 0

    // 2. 初始化/赋值
    *p = 100 // 修改指针指向的值

    fmt.Println(*p) // 输出 100
}
```

---

## 3. Slice 和 Map - 最麻烦的情况

这也是为什么不推荐对 Slice/Map 使用 `new` 的原因。如果你对 Slice 或 Map 用了 `new`，你得到的是一个指向 `nil` 的指针。**你必须先创建一个非 nil 的实际对象（用 make 或字面量），然后把它赋值给那个指针。**

这叫做"脱裤子放屁"——多此一举。

### 如果强行用 new 初始化 Map：

```go
func main() {
    // 1. mp 是 *map[string]int
    mp := new(map[string]int)

    // *mp 目前是 nil。
    // (*mp)["a"] = 1 // 崩溃！不能往 nil map 写数据

    // 2. 初始化：必须再次调用 make，并赋值给 *mp
    *mp = make(map[string]int) // 这里才是真正分配 map 内部结构的地方

    // 3. 现在才能用
    (*mp)["a"] = 1
}
```

### 如果强行用 new 初始化 Slice：

```go
func main() {
    // sp 是 *[]int
    sp := new([]int)

    // *sp 目前是 nil

    // 初始化：赋值一个由 make 创建的切片，或者字面量
    *sp = make([]int, 0, 10)
    // 或者 *sp = []int{1, 2, 3}

    // 现在可以用 append 了
    *sp = append(*sp, 1)
}
```

---

## 总结对比

| 类型 | 使用 new 之后 | 怎么初始化/赋值 | 代码示例 |
| --- | --- | --- | --- |
| **Struct** | 得到结构体指针 | 直接用 `.` 赋值字段 | `u.Name = "A"` |
| **int/string** | 得到值指针 | 用 `*` 解引用赋值 | `*i = 10` |
| **Map/Slice** | 得到指向 `nil` 的指针 | **必须再 make 一次**并赋值给指针 | `*m = make(...)` |

---

## 最佳实践建议

既然 `new` 后面如果要赋初值（特别是 struct）还得一行一行写 `u.Field = val`，所以 Go 社区更推荐使用 **Struct 字面量**，因为它可以在创建时直接初始化：

```go
// 推荐写法：一步到位
u := &User{
    Name: "Gemini",
    Age:  18,
}
```

这比 `new` + 赋值 更紧凑、更直观。

---

## 相关阅读

- [make函数详解](./make-function.md) - 了解 `make` 与 `new` 的区别
- [nil 切片 vs 空切片](./nil-vs-empty-slice.md) - 理解 nil 和空切片在内存中的差异
