---
sidebar_position: 10
title: Swift 核心语法速查
tags: [swift, ios, syntax]
---

# Swift 核心语法速查

这份文档涵盖了基础语法，还特别强调了现代 Swift（5.5+）的特性，以及在实际工程开发（尤其是配合 SwiftUI 和架构设计时）最常用的高级特性。

## 1. 基础声明与类型 (Basics)

Swift 是强类型语言，拥有极其强大的类型推断能力。

### 常量与变量

```swift
let maximumLogins = 10     // 常量 (不可变，推荐默认使用)
var currentLogins = 0      // 变量 (可变)
```

### 类型推断与显式声明

```swift
var name = "郝"            // 自动推断为 String
var age: Int = 28         // 显式声明类型
var balance: Double = 0.0 // 明确为 Double 而不是 Int
```

### 字符串插值

```swift
let message = "你好，\(name)，你今年 \(age) 岁。"
```

## 2. 核心灵魂：可选类型 (Optionals)

用来安全处理 `nil`，是 Swift 最重要的安全机制。

### 声明可选类型 (?)

```swift
var nickname: String? = "小明"
var avatarURL: String? = nil // 允许为空
```

### 安全解包 (if let / guard let)

```swift
// 方式一：if let (局部作用域内使用)
if let nickname {
    print("昵称是：\(nickname)")
}

// 方式二：guard let (提前退出，保持代码扁平，后端最爱的参数校验风格)
func updateProfile(url: String?) {
    guard let validURL = url else {
        print("URL 为空，退出执行")
        return
    }
    print("开始下载: \(validURL)") // validURL 在函数后续全局可用
}
```

### 空值合并运算符 (??)

如果为空，则提供一个默认值。

```swift
let finalName = nickname ?? "默认用户"
```

### 强制解包 (!) —— 危险

如果确定 100% 不为空才使用，否则一旦为空直接崩溃。

```swift
let length = nickname!.count
```

## 3. 控制流 (Control Flow)

没有多余的括号，且条件必须是严格的布尔值。

### If / Else

```swift
if age >= 18 {
    print("成年")
} else {
    print("未成年")
}
```

### Switch (必须穷尽，不需要 break)

```swift
let status = 200
switch status {
case 200...299: // 支持区间匹配
    print("请求成功")
case 404, 500:  // 支持多条件匹配
    print("请求失败")
default:
    print("未知状态")
}
```

### For-In 循环

```swift
let names = ["Anna", "Alex", "Brian"]
for name in names { ... }

// 遍历字典
let ages = ["Anna": 20, "Alex": 25]
for (name, age) in ages { ... }

// 遍历区间
for i in 0..<5 { print(i) } // 0,1,2,3,4
for i in 1...3 { print(i) } // 1,2,3
```

## 4. 函数与闭包 (Functions & Closures)

极其注重 API 的自然语言可读性。

### 双命名参数 (外部标签 & 内部变量)

```swift
// to 和 with 是外部标签，target 和 config 是内部变量
func connect(to target: String, with config: String) -> Bool {
    print("Connecting \(target) via \(config)")
    return true
}

connect(to: "192.168.1.1", with: "TCP")

// 忽略外部标签用 _
func add(_ a: Int, _ b: Int) -> Int { return a + b }
add(3, 5)
```

### 闭包 (Closures) & 尾随闭包

闭包是自包含的代码块（类似匿名函数 / Lambda）。

```swift
// 定义一个接收闭包作为参数的函数
func fetchData(completion: (String) -> Void) {
    completion("数据返回")
}

// 尾随闭包写法 (SwiftUI 的基石)
// 如果闭包是函数的最后一个参数，可以直接写在括号外面
fetchData { result in
    print("拿到结果：\(result)")
}
```

## 5. 自定义数据类型 (Data Structures)

Swift 拥有极其强大的类型系统，不仅有类和结构体，枚举更是"一等公民"。

### Struct (结构体) —— 默认选择，值类型

```swift
struct User {
    var id: Int
    var name: String
}
// 自动获得一个逐一成员构造器
var user = User(id: 1, name: "郝")
```

### Class (类) —— 共享状态，引用类型

```swift
class NetworkManager {
    var isConnected = false
    init() { } // 必须手动写构造器或给默认值
}
```

### Enum (枚举) —— 极其强大，支持关联值 (Payload)

在 Swift 中，枚举不仅能表示状态，还能携带数据（常用于状态机、TCA 架构中的 Action、错误定义）。

```swift
enum APIResponse {
    case success(data: String) // 携带成功的数据
    case failure(code: Int)    // 携带错误码
    case loading
}

let response = APIResponse.success(data: "JSON 内容")

// 使用 switch 提取关联值
switch response {
case .success(let data):
    print("解析数据：\(data)")
case .failure(let code):
    print("报错，错误码：\(code)")
case .loading:
    print("加载中...")
}
```

## 6. 协议与扩展 (Protocols & Extensions)

"面向协议编程"是 Swift 的核心范式（类似 Go 的 Interface 组合）。

### Protocol (协议)

定义必须实现的方法和属性。

```swift
protocol Drawable {
    func draw()
}
```

### Extension (扩展)

可以为任何已有的类型（哪怕是系统的 `String`、`Int` 或第三方库的类型）添加新方法，或者让它们遵守新的协议。

```swift
extension String {
    var isEmail: Bool {
        return self.contains("@")
    }
}
print("test@abc.com".isEmail) // true
```

## 7. 错误处理 (Error Handling)

不同于 Go 的返回值判断，Swift 使用 `do-try-catch` 机制，强制处理错误。

```swift
// 1. 定义错误 (遵守 Error 协议)
enum NetworkError: Error {
    case timeout
    case notFound
}

// 2. 抛出错误的函数 (使用 throws 关键字标记)
func fetch() throws -> String {
    throw NetworkError.timeout
}

// 3. 捕获并处理错误
do {
    let data = try fetch()
    print(data)
} catch NetworkError.timeout {
    print("超时了")
} catch {
    print("其他错误: \(error)")
}

// 便捷写法：如果不在乎具体错误，只想把错误变成 nil (返回可选类型)
let safeData = try? fetch() // safeData 是 String?
```

## 8. 现代并发 (Concurrency)

基于协程调度的结构化并发（Swift 5.5+）。

### async / await

```swift
func downloadImage() async throws -> String {
    try await Task.sleep(for: .seconds(1)) // 挂起，不阻塞线程
    return "图片数据"
}
```

### Task (启动协程)

```swift
Task {
    do {
        let img = try await downloadImage()
    } catch {
        print("下载失败")
    }
}
```

### Actor (线程安全模型)

自动解决多线程数据竞争，内部状态同一时间仅允许一个任务访问。

```swift
actor DataStore {
    var cache: [String] = []
    func save(_ item: String) { cache.append(item) }
}
// 外部调用必须使用 await
```
