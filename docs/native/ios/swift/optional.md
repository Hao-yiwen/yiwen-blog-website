---
sidebar_position: 4
---

# 可选类型

## 介绍

因为`swift`是静态编译型语言，所以在可选类型的使用上和js有很大却别，并不会直接为null，而是需要更为精确和安全的管理。

## 示例

```swift
class NamedShape {
    var numberOfSides: Int = 0
    var name: String

    init(name: String) {
        self.name = name
    }

    func simpleDescription() -> String {
        return "A shape with \(numberOfSides) sides."
    }
}

class Square: NamedShape {
    var sideLength: Double

    init(sideLength: Double, name: String) {
        self.sideLength = sideLength
        super.init(name: name)
        numberOfSides = 4
    }

    func area() ->  Double {
        return sideLength * sideLength
    }

    override func simpleDescription() -> String {
        return "A square with sides of length \(sideLength)."
    }
}
let test = Square(sideLength: 5.2, name: "my test square")
print(test.area())
print(test.simpleDescription())
print(test.sideLength)

print("======")
let optionalSquare: Square? = Square(sideLength: 2.5, name: "optional square")
let sideLength = optionalSquare?.sideLength
if let side = sideLength { // 判断sideLength
    let result = side + 3
    print(result)
} else {
    print("sideLength is nil")
}
print("======")
print(sideLength! + 3)  // 指定sideLength不为空
print("=======")
let optionalSquare: Square? = Square(sideLength: 2.5, name: "optional square")
let sideLength = optionalSquare?.sideLength ?? 0 // 如果 sideLength 为 nil，则使用默认值 0
print(sideLength + 3)
```
