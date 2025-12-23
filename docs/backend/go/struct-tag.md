---
sidebar_position: 6
title: Struct Tag
tags: [go, struct, json, gorm, 结构体标签]
---

# Go Struct Tag（结构体标签）详解

在 Go 语言中，Struct Tag 是一个非常有特色的功能。你可以把它想象成贴在每个字段背后的**"便利贴"**或**"说明书"**。

## 1. 它是干什么用的？

Go 语言本身并不去读这些标签，但是**其他的库（包）会去读**。

例如，`json:"id"` 是给 Go 标准库 `encoding/json` 看的。它告诉 JSON 转换器："嘿，虽然我在 Go 代码里叫 `ID`（大写），但在转成 JSON 的时候，请把我改名叫 `id`（小写）。"

## 2. 为什么要这么做？

主要有两个原因：**大小写规则冲突** 和 **元数据控制**。

### 原因 A：解决命名风格冲突

- **Go 的规则**：如果你想让一个字段能被外部访问（比如能被转成 JSON），首字母必须**大写**（如 `Balance`）
- **前端/API 的规则**：JSON 数据通常习惯用**小写**或下划线命名（如 `balance` 或 `user_balance`）

如果没有 Tag，Go 默认会直接用字段名：

```json
{"ID": "123", "Balance": 100.0}  // 前端可能不想要大写
```

加上 Tag `` `json:"id"` `` 后，Go 就会自动帮你改名：

```json
{"id": "123", "balance": 100.0}  // 这才是前端想要的
```

### 原因 B：控制特殊行为

除了改名，Tag 还能控制很多行为。这里有几个最常用的 `json` 标签指令：

| 标签写法 | 含义 | 效果示例 |
| --- | --- | --- |
| `` `json:"age"` `` | **重命名** | Go 里叫 `Age`，JSON 里变成 `"age"` |
| `` `json:"-"` `` | **忽略/隐藏** | 无论字段里有什么值，**永远不输出到 JSON**（比如密码字段） |
| `` `json:"age,omitempty"` `` | **为空省略** | 如果字段是"零值"（0, "", nil），则**直接不在 JSON 里出现** |
| `` `json:",string"` `` | **转字符串** | 把数值类型强制转成字符串输出（如 `"100"` 而不是 `100`） |

## 3. 不止是 JSON

这个 Tag 机制是通用的，很多第三方库都在用。你经常会看到一个字段后面挂着一长串标签，用空格隔开。

例如，一个 Web 开发中常见的 User 结构体：

```go
type User struct {
    // JSON 库用：转成 JSON 时叫 "id"
    // GORM 库用：数据库里是主键 (primaryKey)
    ID int `json:"id" gorm:"primaryKey"`

    // JSON 库用：转成 "username"
    // Form 库用：处理网页表单提交时，对应 input name="user"
    // Validate 库用：必须包含字母 (alpha)
    Name string `json:"username" form:"user" validate:"alpha"`

    // JSON 库用：转成 "password"，但是在输出 JSON 时总是忽略（为了安全）
    Password string `json:"-"`
}
```

## 4. 常见的 Tag 类型

| Tag 前缀 | 用途 | 常用库 |
| --- | --- | --- |
| `json` | JSON 序列化/反序列化 | `encoding/json` |
| `xml` | XML 序列化/反序列化 | `encoding/xml` |
| `gorm` | 数据库 ORM 映射 | GORM |
| `db` | 数据库字段映射 | sqlx |
| `form` | 表单绑定 | Gin, Echo |
| `validate` | 数据验证 | go-playground/validator |
| `yaml` | YAML 序列化 | gopkg.in/yaml.v3 |
| `mapstructure` | Map 到 Struct 转换 | mapstructure |

## 5. 语法注意事项

写 Tag 的时候非常严格，**冒号前后绝对不能有空格**！

```go
// ✅ 正确
ID int `json:"id"`

// ❌ 错误（多了空格，无法识别）
ID int `json: "id"`
```

多个 Tag 之间用**空格**分隔：

```go
// ✅ 正确
Name string `json:"name" gorm:"column:user_name"`
```

## 6. omitempty 实战

`omitempty` 对减少网络传输流量很有用：

```go
package main

import (
    "encoding/json"
    "fmt"
)

type Account struct {
    ID      string   `json:"id"`
    Balance *float64 `json:"balance,omitempty"`
    Name    string   `json:"name,omitempty"`
}

func main() {
    // 有值的情况
    balance := 100.0
    acc1 := Account{ID: "001", Balance: &balance, Name: "Alice"}

    // 空值的情况
    acc2 := Account{ID: "002"}

    json1, _ := json.Marshal(acc1)
    json2, _ := json.Marshal(acc2)

    fmt.Println(string(json1))
    // 输出: {"id":"001","balance":100,"name":"Alice"}

    fmt.Println(string(json2))
    // 输出: {"id":"002"}  <- balance 和 name 字段直接消失了
}
```

## 7. 如何读取 Tag

你也可以通过反射（reflect）来读取 Tag：

```go
package main

import (
    "fmt"
    "reflect"
)

type User struct {
    Name string `json:"name" validate:"required"`
}

func main() {
    t := reflect.TypeOf(User{})
    field, _ := t.FieldByName("Name")

    fmt.Println(field.Tag)              // json:"name" validate:"required"
    fmt.Println(field.Tag.Get("json"))  // name
    fmt.Println(field.Tag.Get("validate")) // required
}
```

## 总结

Struct Tag 就是**给代码看的代码**：

- 你在定义数据结构
- Tag 在告诉转换器（JSON、数据库、表单验证器）如何处理这个数据

这是 Go 语言中实现**元编程**和**声明式配置**的重要机制。
