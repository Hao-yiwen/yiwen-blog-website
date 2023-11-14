---
sidebar_position: 1
---

# SwiftUI

## 介绍

在`Ios`的`UI`的开发选择中，我选择`swiftUI`作为第一个学习的`UI`库，自2019年`SwiftUI`诞生以来，经过4年的迭代，`SwiftUI`日趋成熟，并且作为`Apple`跨平台开发视图库，可以有效提升效率。

## 数据类型

### 基本数据类型

-   Int, UInt: 用于整数。Int 是有符号整数类型，UInt 是无符号整数类型。
-   Float, Double: 浮点数类型。Float 表示单精度浮点数，而 Double 表示双精度浮点数。
-   Bool: 布尔类型，表示逻辑上的真（true）或假（false）。

```swift
var anInteger: Int = 5
var aUnsignedInteger: UInt = 7
var aFloat: Float = 3.14
var aDouble: Double = 2.71828
var aBoolean: Bool = true
```

### 字符和字符串类型

-   Character: 用于单个字符。
-   String: 用于文本数据。

```swift
var aCharacter: Character = "A"
var aString: String = "Hello, Swift!"
```

### 集合类型

-   Array: 有序的值的集合。
-   Set: 无序的唯一值的集合。
-   Dictionary: 无序的键值对的集合。

```swift
var anArray: [Int] = [1, 2, 3]
var aSet: Set<Int> = [1, 2, 3]
var aDictionary: [String: Int] = ["one": 1, "two": 2, "three": 3]
```

### 可选类型

-   Optional: 用于处理值可能缺失的情况。它可以包含一个值或者 nil。

```swift
var optionalInt: Int? = 5
optionalInt = nil
```

### 元组

-   Tuple: 用于创建和传递一组数据。元组内的值可以是任何类型，并不要求是相同类型。

```swift
var aTuple = (404, "Not Found")
var anotherTuple: (code: Int, message: String) = (200, "OK")
```

### 枚举类型

-   Enum: 用于定义一组相关的值，并使你能够以类型安全的方式处理这些值。

```swift
enum Direction {
    case north
    case south
    case east
    case west
}
var aDirection = Direction.north
```

### 类型别名

-   Type Alias: 允许你定义现有类型的另一个名称。

```swift
typealias StringDictionary = [String: String]
var someDictionary: StringDictionary = ["key1": "value1", "key2": "value2"]
```

### 函数类型

-   Function Type: 表示具有特定参数类型和返回类型的函数。

```swift
func addTwoInts(_ a: Int, _ b: Int) -> Int {
    return a + b
}
var mathFunction: (Int, Int) -> Int = addTwoInts
```

### 闭包

-   Closure: 无名函数，可以捕获和存储其所在上下文中的常量和变量的引用。

```swift
let square: (Int) -> Int = { number in
    return number * number
}
let squaredNumber = square(3)
```

### 类和结构体

-   Class: 引用类型，可以继承另一个类。
-   Struct: 值类型，用于封装数据和相关行为

```swift
class MyClass {
    var name: String = "MyClass"
}
struct MyStruct {
    var name: String = "MyStruct"
}

var aClassInstance = MyClass()
var aStructInstance = MyStruct()
```

### Any 和 AnyObject

-   Any: 可以表示任何类型的实例，包括函数类型
-   AnyObject: 可以表示任何类类型的实例。

```swift
var anyVariable: Any = 4
anyVariable = "Changed to a string"
anyVariable = { print("I'm a closure!") }

class SomeClass {}
var anyObjectVariable: AnyObject = SomeClass()
```

### Void

-   Void: 表示没有返回值的函数的返回类型，等同于空的元组 ()。

```swift
func doNothing() -> Void {
    // This function does not return a value
}
// Or simply
func doNothingAgain() {
    // Implicitly returns Void
}
```

## 函数闭包

在 Swift 中，函数和闭包实际上是非常相似的概念，因此它们的声明方式也相似。这种设计反映了 Swift 中函数作为一等公民（first-class citizens）的特性，意味着函数可以像任何其他类型一样被传递和赋值。闭包本质上是匿名函数，它们也遵循相同的规则。

### 函数和闭包的相似性

1. 类型签名：函数和闭包都有类型签名，即它们的参数和返回类型。比如，() -> Int 表示一个没有参数并返回 Int 类型的函数或闭包。

2. 作为变量：函数和闭包都可以赋值给变量或常量。这意味着您可以将一个函数或闭包存储在变量中，并在需要时调用。

```swift title="函数"
func someFunction() -> Int {
    return 10
}
var a: () -> Int = someFunction
```

```swift title="闭包"
var b: () -> Int = {
    return 20
}
```

### 使用上的区别

1. 闭包可以捕获值：闭包可以捕获并存储其定义上下文中的常量和变量引用，而普通函数则不能。
2. 语法简洁性：闭包有更灵活的语法，允许简写，比如省略参数名、使用隐式返回等。
3. 匿名性：闭包可以是匿名的，不需要命名，而函数总是有名称。

### 总结

Swift 中函数和闭包的类型声明相似是因为它们本质上是相同的概念：自包含的代码块，可以在代码中传递和调用。这种设计让 Swift 代码更加灵活和表达性强，允许更高阶的编程模式，如函数式编程风格。
