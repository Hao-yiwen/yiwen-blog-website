---
sidebar_position: 9
title: 核心库实战指南
tags: [go, 标准库, fmt, bufio, strings, strconv, json, os, time, sync, context, net/http]
---

# Go 核心库实战指南

Go 语言的学习核心就是"多写代码"。本文将 Go 语言最常用的核心库拆解，并为每一个部分提供**可运行的、地道的（Idiomatic）代码示例**。

---

## 1. 基础输入输出 (`fmt`, `bufio`)

**场景**：格式化打印日志，或者从控制台/文件高效读取数据。

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	// --- fmt: 格式化 ---
	name := "Gopher"
	age := 10
	// %s: 字符串, %d: 整数, %.2f: 浮点数, %v: 通用值, %+v: 带字段名的结构体
	fmt.Printf("Hello, %s! You are %d years old.\n", name, age)

	// Sprintf 用于生成字符串而不打印
	msg := fmt.Sprintf("User[%s] is active", name)
	fmt.Println(msg)

	// --- bufio: 带缓冲读取 (例如从键盘读取输入) ---
	fmt.Print("Please enter text: ")
	reader := bufio.NewReader(os.Stdin)
	// 读取直到遇到换行符
	input, _ := reader.ReadString('\n')
	fmt.Printf("You entered: %s", strings.TrimSpace(input))
}
```

---

## 2. 字符串与转换 (`strings`, `strconv`)

**场景**：处理文本数据，将字符串转为数字进行计算。

```go
package main

import (
	"fmt"
	"strconv"
	"strings"
)

func main() {
	// --- strings: 字符串处理 ---
	data := "apple,banana,orange"
	// 分割
	parts := strings.Split(data, ",")
	fmt.Println(parts) // [apple banana orange]

	// 包含检查
	if strings.Contains(data, "banana") {
		fmt.Println("Found banana!")
	}

	// --- strconv: 类型转换 ---
	numStr := "128"
	// 字符串 转 int (常用)
	num, err := strconv.Atoi(numStr)
	if err != nil {
		fmt.Println("Conversion error:", err)
	} else {
		fmt.Printf("Converted: %d, Type: %T\n", num, num)
	}

	// int 转 字符串
	str := strconv.Itoa(42)
	fmt.Println("String:", str)
}
```

---

## 3. JSON 数据处理 (`encoding/json`)

**场景**：Web 开发中最常用的数据交换格式。

```go
package main

import (
	"encoding/json"
	"fmt"
)

// 定义结构体，使用 `json:"..."` 标签控制字段名
type User struct {
	ID       int    `json:"user_id"`
	Username string `json:"username"`
	IsAdmin  bool   `json:"is_admin,omitempty"` // 如果是 false 则不输出
	Password string `json:"-"`                  // 永远不输出到 JSON
}

func main() {
	// 1. Marshal: 结构体 -> JSON 字符串
	u := User{ID: 1, Username: "admin_01", IsAdmin: true, Password: "secret"}
	jsonData, _ := json.Marshal(u)
	fmt.Println(string(jsonData))
	// 输出: {"user_id":1,"username":"admin_01","is_admin":true}

	// 2. Unmarshal: JSON 字符串 -> 结构体
	jsonStr := `{"user_id":2, "username":"guest"}`
	var u2 User
	if err := json.Unmarshal([]byte(jsonStr), &u2); err == nil {
		fmt.Printf("Parsed User: %+v\n", u2)
	}
}
```

---

## 4. 文件与路径 (`os`, `path/filepath`)

**场景**：跨平台地处理文件路径，读写文件。

```go
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	// --- filepath: 安全拼接路径 ---
	// 自动处理 Windows (\) 和 Linux (/) 的差异
	dir := "data"
	fileName := "config.txt"
	fullPath := filepath.Join(dir, fileName)
	fmt.Println("Path:", fullPath)

	// --- os: 写文件 ---
	content := []byte("Hello, Go File System!")
	// 0644 是文件权限 (rw-r--r--)
	err := os.WriteFile(fileName, content, 0644)
	if err != nil {
		fmt.Println("Write error:", err)
		return
	}

	// --- os: 读文件 ---
	readData, err := os.ReadFile(fileName)
	if err == nil {
		fmt.Println("File content:", string(readData))
	}

	// 清理文件
	os.Remove(fileName)
}
```

---

## 5. 时间处理 (`time`)

**场景**：定时任务、计算耗时、日期格式化。

```go
package main

import (
	"fmt"
	"time"
)

func main() {
	now := time.Now()

	// --- 格式化 ---
	// Go 的特殊之处：必须用固定的参考时间 "2006-01-02 15:04:05"
	fmt.Println("Current:", now.Format("2006-01-02 15:04:05"))

	// --- 时间计算 ---
	oneHourLater := now.Add(1 * time.Hour)
	fmt.Println("In 1 hour:", oneHourLater)

	// --- 时间段与比较 ---
	start := time.Now()
	time.Sleep(100 * time.Millisecond) // 模拟耗时
	elapsed := time.Since(start)
	fmt.Printf("Operation took: %v\n", elapsed)
}
```

---

## 6. 并发控制 (`sync`, `context`)

**场景**：这是 Go 的杀手锏。如果你需要并发执行任务，或者控制超时，必须掌握这里。

```go
package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

func main() {
	// --- sync.WaitGroup: 等待一组协程完成 ---
	var wg sync.WaitGroup

	for i := 1; i <= 3; i++ {
		wg.Add(1) // 计数器 +1
		go func(id int) {
			defer wg.Done() // 函数结束时计数器 -1
			fmt.Printf("Worker %d starting\n", id)
			time.Sleep(time.Millisecond * 500)
		}(i)
	}

	wg.Wait() // 阻塞直到计数器归零
	fmt.Println("All workers done.")

	// --- context: 超时控制 ---
	// 创建一个 1秒后自动超时的 context
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	select {
	case <-time.After(2 * time.Second): // 模拟一个耗时2秒的操作
		fmt.Println("Task finished (slow)")
	case <-ctx.Done(): // 监听 context 是否超时
		fmt.Println("Task timed out:", ctx.Err()) // 输出 context deadline exceeded
	}
}
```

---

## 7. 网络编程 (`net/http`)

**场景**：用极少的代码启动一个 Web 服务器。

```go
package main

import (
	"fmt"
	"net/http"
)

// 处理函数
func helloHandler(w http.ResponseWriter, r *http.Request) {
	// 解析查询参数 ?name=Go
	name := r.URL.Query().Get("name")
	if name == "" {
		name = "Stranger"
	}
	// 写入响应
	fmt.Fprintf(w, "Hello, %s!", name)
}

func main() {
	// 注册路由
	http.HandleFunc("/", helloHandler)

	fmt.Println("Server starting on :8080...")
	// 启动监听 (这一步会阻塞)
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		fmt.Println("Server failed:", err)
	}
}
```

*运行后，访问浏览器 `http://localhost:8080/?name=Master` 即可看到效果。*

---

## 总结

| 库 | 核心用途 | 关键函数/类型 |
| --- | --- | --- |
| `fmt` | 格式化输出 | `Printf`, `Sprintf`, `Println` |
| `bufio` | 带缓冲 I/O | `NewReader`, `ReadString` |
| `strings` | 字符串操作 | `Split`, `Contains`, `TrimSpace` |
| `strconv` | 字符串转换 | `Atoi`, `Itoa`, `ParseInt` |
| `encoding/json` | JSON 序列化 | `Marshal`, `Unmarshal` |
| `os` | 文件/环境操作 | `ReadFile`, `WriteFile`, `Remove` |
| `path/filepath` | 跨平台路径 | `Join`, `Dir`, `Base` |
| `time` | 时间处理 | `Now`, `Format`, `Add`, `Since` |
| `sync` | 并发同步 | `WaitGroup`, `Mutex`, `Once` |
| `context` | 超时/取消控制 | `WithTimeout`, `WithCancel` |
| `net/http` | HTTP 服务 | `HandleFunc`, `ListenAndServe` |

掌握这些核心库，你就能覆盖 Go 语言日常开发中 80% 以上的场景。
