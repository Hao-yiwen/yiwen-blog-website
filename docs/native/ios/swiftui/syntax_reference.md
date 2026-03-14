---
sidebar_position: 10
title: SwiftUI 核心语法速查
tags: [swift, swiftui, ios, syntax]
---

# SwiftUI 核心语法速查

有了 Swift 语言基础（结构体、闭包、属性包装器、并发），再来看 SwiftUI 的语法，它是一套极其优雅的声明式 UI 框架。SwiftUI 和 React 的组件化思想高度一致：**UI = f(State)**。

## 1. 视图的基本解剖 (The Anatomy of a View)

在 SwiftUI 中，所有的 UI 组件都是轻量级的 `struct`，并且必须遵守 `View` 协议，实现一个计算属性 `body`。

```swift
import SwiftUI

// 1. 必须是 struct，遵守 View 协议
struct HealthDashboardView: View {
    // 2. body 前面的 some View 叫做"不透明返回类型"
    // 意思是我保证返回一个 View，但具体是什么类型由编译器去推断
    var body: some View {
        // 3. 只能返回一个单一的顶级视图容器
        Text("今日健康数据")
    }
}
```

## 2. 布局三大金刚 (Stack Layouts)

SwiftUI 抛弃了复杂的约束（Auto Layout），主要靠三种 Stack 来堆叠界面，极其类似前端的 Flexbox。

- **VStack** (垂直排布)：类似 `flex-direction: column`
- **HStack** (水平排布)：类似 `flex-direction: row`
- **ZStack** (Z轴层叠)：类似 `position: absolute`，后面的元素覆盖在前面的元素之上

```swift
var body: some View {
    VStack(alignment: .leading, spacing: 16) { // 垂直排列，左对齐，间距 16
        Text("步数")
            .font(.headline)

        HStack { // 内部水平排列
            Image(systemName: "figure.walk") // 系统内置图标
                .foregroundColor(.green)
            Text("8,432 步")
                .font(.title)
                .bold()

            Spacer() // 弹簧占位符：把前面的元素顶到左边，后面的顶到右边

            Text("目标: 10k")
                .foregroundColor(.gray)
        }

        ZStack { // 层叠：背景在下，文字在上
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.blue.opacity(0.1))
                .frame(height: 50)
            Text("完成度: 84%")
                .foregroundColor(.blue)
        }
    }
    .padding() // 给整个 VStack 加一圈内边距
}
```

## 3. 修饰符 (Modifiers) 与链式调用

SwiftUI 通过链式调用来改变视图的样式。

**极其重要的概念：修饰符的顺序严格影响最终结果！** 每次调用修饰符，实际上是把原来的 View 包装进了一个新的 View 里。

```swift
Text("心率 72 BPM")
    .padding()               // 先加一圈内边距 (撑大盒子)
    .background(Color.red)   // 给撑大后的盒子涂上红色背景
    .cornerRadius(8)         // 给红色的盒子切圆角

// 对比下面这个：
Text("心率 72 BPM")
    .background(Color.red)   // 先涂红色背景 (只有文字那么大)
    .padding()               // 再加内边距 (外面多了一圈透明的边)
```

## 4. 状态与数据双向绑定 (State & Binding)

这是 SwiftUI 数据驱动的核心。视图只是一份"图纸"，状态变了，视图自动销毁重建。

- **@State**：用于视图内部的私有状态（类似 React 的 `useState`）
- **$ (美元符号)**：用于生成双向绑定引用（`Binding`），传给子视图或输入框

```swift
struct InputView: View {
    // 声明状态
    @State private var weight: String = ""
    @State private var isSaved: Bool = false

    var body: some View {
        VStack {
            // TextField 需要双向绑定，所以用 $weight
            TextField("请输入体重 (kg)", text: $weight)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.decimalPad)

            Button("保存") {
                // 点击按钮修改状态，不需要加 $，直接操作值
                isSaved = true
                print("保存的值是：\(weight)")
            }

            if isSaved {
                Text("已保存！")
                    .foregroundColor(.green)
            }
        }
        .padding()
    }
}
```

## 5. 列表与循环 (Lists & ForEach)

当有一组数据需要渲染时，使用 `List`（自带原生滚动和分割线）和 `ForEach`。

```swift
struct Record: Identifiable { // 必须遵守 Identifiable，或者在 ForEach 里指定 id
    let id = UUID()
    let date: String
    let value: Int
}

struct HistoryView: View {
    let records = [
        Record(date: "周一", value: 120),
        Record(date: "周二", value: 115)
    ]

    var body: some View {
        // List 会自动变成一个原生可滚动的列表
        List {
            ForEach(records) { record in
                HStack {
                    Text(record.date)
                    Spacer()
                    Text("\(record.value) mmHg")
                }
            }
        }
    }
}
```

## 6. 导航 (Navigation)

用于页面之间的跳转。通常在最外层套一个 `NavigationStack`。

```swift
struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack {
                // NavigationLink 相当于前端的 <Link> 或 <a> 标签
                NavigationLink(destination: DetailView()) {
                    Text("查看详细报告")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
            }
            .navigationTitle("健康主页") // 设置当前页面的导航栏标题
        }
    }
}

struct DetailView: View {
    var body: some View {
        Text("这是详情页")
            .navigationTitle("报告详情")
            .navigationBarTitleDisplayMode(.inline) // 让标题变小居中
    }
}
```

## 7. 生命周期与并发入口

SwiftUI 视图没有传统的 `viewDidLoad` 等复杂的生命周期，主要靠修饰符来触发。

- **.onAppear**：视图出现时执行（同步代码）
- **.task**：视图出现时执行（异步代码的最佳位置，视图销毁时自动取消请求）

```swift
Text(userName)
    .task {
        // 这里天然就是异步环境，可以直接写 await
        // 非常适合在这里向后端发起 API 请求，或者读取系统底层的数据
        let data = await fetchNetworkData()
        userName = data
    }
```

## 8. 属性包装器速查 (@xxx)

在概念上可以把它们当成装饰器来理解：它们都是在不改变原有业务逻辑的情况下，给变量或类"悄悄"注入了额外的超能力。在 Swift 的官方术语里，主要分为两类：

- **属性包装器 (Property Wrappers)**：比如 `@State`、`@Binding`
- **宏 (Macros)**：比如 iOS 17 引入的 `@Observable`

### 第一梯队：视图内部的局部状态 (类似 React 的 useState)

#### @State (状态)

让一个原本不可变的 `struct` 里的变量变得可修改，只要这个变量一变，UI 就自动重新渲染。

- 适用场景：只在当前视图内部使用的简单数据（比如开关状态、输入框文字）
- 规范：官方强烈建议永远加上 `private`

```swift
struct ContentView: View {
    @State private var isOn: Bool = false // 注入了"监听和刷新UI"的超能力
    var body: some View {
        Toggle("开关", isOn: $isOn)
    }
}
```

#### @Binding (绑定)

子视图想要修改父视图的 `@State` 变量，就需要用 `@Binding` 接收。它相当于传了一个"指针"或者引用的通道。

- 适用场景：封装独立组件时，把状态控制权交还给父级

```swift
struct CustomSwitch: View {
    @Binding var isOn: Bool // 接收父视图传来的 $isOn
    var body: some View {
        Button(isOn ? "关" : "开") {
            isOn.toggle() // 这里修改，父视图的 @State 也会跟着变！
        }
    }
}
```

### 第二梯队：复杂的业务逻辑与共享状态 (iOS 17+)

当把业务逻辑（比如网络请求、解析数据）抽离到独立的 `class` 中时，使用以下包装器。

#### @Observable (可观察的宏)

加在 `class` 前面，让这个类的所有属性都自动具备"触发 UI 刷新"的能力。这是 iOS 17 极度精简的新语法，完美替代了以前复杂的 `@StateObject` 和 `@Published`。

- 适用场景：ViewModel、数据管理器、网络请求层

```swift
@Observable
class HealthManager {
    var stepCount: Int = 0 // UI 可以直接监听它
    var isFetching: Bool = false
}

struct DashboardView: View {
    @State private var manager = HealthManager()
    // ...
}
```

#### @Bindable (可绑定对象)

当把 `@Observable` 的对象传给子视图，且子视图里面的输入框想要直接修改对象里的属性时，需要用它包装一下，这样才能使用 `$` 符号。

```swift
struct EditProfileView: View {
    @Bindable var userManager: UserManager
    var body: some View {
        // 使用 $userManager.name 实现双向绑定
        TextField("姓名", text: $userManager.name)
    }
}
```

### 第三梯队：全局环境注入 (类似 React 的 Context API)

#### @Environment (环境变量)

从系统的全局"环境池"里捞数据。比如获取当前系统的深色/浅色模式、屏幕尺寸、或者全局注入的路由对象。彻底解决了属性需要一层层往下传（Prop Drilling）的痛点。

```swift
struct DetailView: View {
    // 捞取系统自带的环境变量：控制页面返回
    @Environment(\.dismiss) private var dismiss
    // 捞取系统自带的颜色模式
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Button("返回上一页") {
            dismiss() // 调用系统方法关闭当前页面
        }
    }
}
```

#### @AppStorage (应用本地存储)

对系统 `UserDefaults`（类似于前端的 `localStorage`）的极简包装。数据不仅会持久化存在手机硬盘里，而且改动时还会自动刷新 UI。

- 适用场景：保存用户的简单偏好设置（比如：是否首次打开 App、主题颜色设置）

```swift
struct SettingsView: View {
    // 存入硬盘的 key 叫 "isDarkMode"，默认值是 false
    @AppStorage("isDarkMode") private var isDarkMode = false
}
```

### 总结

这些 `@` 包装器会在编译时帮你生成大量的 `get` 和 `set` 拦截代码。当你在代码里执行 `isOn = true` 时，它不仅修改了内存里的值，还在 `set` 拦截器里大喊了一声："值变了，SwiftUI 渲染引擎赶紧重新调用当前 View 的 `body`！"
