---
sidebar_position: 6
---

# 泛型

## 示例

```swift
// func
func makeArray<Item>(repeating item: Item, numberOfTimes: Int) -> [Item] {
    var result: [Item] = []
    for _ in 0..<numberOfTimes {
        result.append(item)
    }
    return result
}
makeArray(repeating: "knock", numberOfTimes: 4)
// enum
// 重新实现 Swift 标准库中的可选类型
enum OptionalValue<Wrapped> {
    case none
    case some(Wrapped)
}
var possibleInteger: OptionalValue<Int> = .none
possibleInteger = .some(100)
// 高级泛型
func anyCommonElements<T: Sequence, U: Sequence>(_ lhs: T, _ rhs: U) -> Bool
    where T.Element: Equatable, T.Element == U.Element
{
    for lhsItem in lhs {
        for rhsItem in rhs {
            if lhsItem == rhsItem {
                return true
            }
        }
    }
    return false
}
anyCommonElements([1, 2, 3], [3])
```

**高级泛型模块解释**

1. Sequence:
   Sequence 是 Swift 标准库中的一个基本协议，它代表了一个元素的序列，这些元素可以被遍历至少一次。数组（Array）、集合（Set）、和字典（Dictionary）都遵循 Sequence 协议，以及许多其他的集合类型。
   Sequence 协议定义了几个方法和属性，但最基本的是 makeIterator() 方法，它返回一个遵循 IteratorProtocol 协议的迭代器对象。通过迭代器，你可以遍历序列的元素。
   通过遵守 Sequence 协议，类型可以提供对其元素的顺序访问，并与 Swift 的 for-in 循环以及许多标准库函数（如 map、filter 和 reduce）交互。

```swift
// 示例: 使用 for-in 循环遍历数组（Array 是一个 Sequence）
for element in [1, 2, 3, 4] {
    print(element)
}
```

2. Equatable:
   Equatable 是 Swift 标准库中的另一个基本协议，它要求类型提供一种方法来测试其实例之间的等价性。
   任何遵循 Equatable 协议的类型必须提供 == 运算符的实现，该运算符比较两个实例并返回一个布尔值，指示它们是否相等。
   大多数基本的 Swift 类型（如 Int、String 和 Double）都遵循 Equatable 协议，许多自定义类型也可以很容易地遵循 Equatable，通常只需通过自动生成的 == 运算符实现。

```swift
// 示例: 检查两个整数是否相等（Int 是一个 Equatable）
let areEqual = (5 == 5)  // true
```
