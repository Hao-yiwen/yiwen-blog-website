---
sidebar_position: 2
---

# @State和@StateObject区别

`@StateObject` 和 `@State` 是` Swift UI` 中用来管理状态的两个不同的属性包装器，它们在用法和目的上有一些关键的区别：

## @State

1. 用途: `@State` 主要用于简单的数据类型，如 `String, Int, Bool` 等。它用来创建一个能够存储值类型的响应式状态变量。
2. 生命周期: 当视图重新创建时，`@State` 变量会保持它们的状态。
3. 所有权: `@State` 是由视图私有的。它不能从其他视图或对象访问。
4. 适用范围: 当你想要在视图内部存储简单的状态时使用。

```swift
struct ContentView: View {
    @State private var name = "John Doe"

    var body: some View {
        Text(name)
    }
}
```

## @stateObject

1. 用途: @StateObject 用于引用类型，特别是遵循 ObservableObject 协议的类。这允许视图订阅这个对象的变化，并在发生变化时重新渲染视图。
2. 生命周期: @StateObject 会在视图首次加载时创建并初始化对象，并在整个视图的生命周期中保持它。
3. 所有权: @StateObject 通常用于视图的根，或者用于视图的所有者。
4. 适用范围: 当你有一个复杂的数据模型或者状态逻辑并且需要在多个视图之间共享时使用。

```swift
class User: ObservableObject {
    @Published var name = "John Doe"
}

struct ContentView: View {
    @StateObject var user = User()

    var body: some View {
        Text(user.name)
    }
}
```

## 总结

-   使用 @State 来存储简单的局部状态。
-   使用 @StateObject 来创建和管理遵循 ObservableObject 的类的实例，并通过多个视图共享这些实例。
    :::info
    注意：当你在父视图中使用 @StateObject 创建了一个对象，并希望在子视图中使用它时，你应该在子视图中使用 @ObservedObject 或者 @EnvironmentObject 来观察和响应这个对象的变化。不要在子视图中再次使用 @StateObject 来创建这个对象的新实例。
    :::

## FAQ

1. 为什么 `@StateObject` 可以用 `@State` 代替（在某些场景下）：

有些情况下，你可能只是简单地需要一个状态的引用，而不需要完全拥有这个状态（例如，你可能在一个父视图中创建了这个状态，并通过 `@ObservedObject` 或 `@EnvironmentObject` 将其传递给子视图）。在这种情况下，你可以使用 `@State` 来代替 `@StateObject`，但这通常不是一个好的做法，因为这可能会引起混淆并可能导致错误。
总的来说，选择使用 `@State, @StateObject, @ObservedObject` 或 `@EnvironmentObject` 应该基于你的具体需求，确保你理解了它们各自的用途和适用场景。在使用引用类型（特别是 ObservableObject）作为状态时，正确使用 @StateObject 和 @ObservedObject 是非常重要的，以确保状态的正确管理和视图的正确更新。
