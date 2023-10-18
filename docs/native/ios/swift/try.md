---
sidebar_position: 7
---

# try和try?还有try!区别

## try

使用 try 关键字前缀的表达式表示该表达式可能会抛出错误。你必须在一个 do-catch 语句中使用它来捕获潜在的错误。

```swift
do {
    let result = try someFunctionThatThrows()
    // Use the result.
} catch {
    print("An error occurred: \(error)")
}
```

## try?

使用 try? 关键字前缀的表达式将结果转换为可选项。如果函数抛出错误，它会返回 nil，否则它会返回结果。

```swift
if let result = try? someFunctionThatThrows() {
    // Use the result.
} else {
    print("An error occurred.")
}
```

## try!:

使用 try! 关键字前缀的表达式表示你确定该函数不会抛出错误。如果函数实际上抛出错误，程序会产生运行时崩溃。

```swift
let result = try! someFunctionThatDefinitelyDoesNotThrow()
// Use the result.
```

⚠️ 警告：应谨慎使用 try!。只有当你确定函数永远不会在任何情况下抛出错误时，才应使用它。

## 总结

-   try 需要配合 do-catch 语句使用，以处理潜在的错误。
-   try? 使你能够处理错误为可选值，如果发生错误，它返回 nil。
-   try! 是一个断言，如果发生错误，它会导致运行时崩溃。
