---
sidebar_position: 5
---

# swift中协议

## protocol接口

```swift
protocol ExampleProtocol {
    var simpleDescription: String { get set }  // 现在同时要求 getter 和 setter
    mutating func adjust()
}
class SimpleClass: ExampleProtocol {
    private var internalDescription = "A simple class"

    var simpleDescription: String {  // swift中的get和set实现
        get {
            return internalDescription
        }
        set {
            internalDescription = newValue
        }
    }

    func adjust() {
        simpleDescription += "  Now adjusted."
    }
}

var simpleClassInstance = SimpleClass()
simpleClassInstance.simpleDescription = "New description"
print(simpleClassInstance.simpleDescription)  // 输出 "New description"
```

## defer

```swift
var fridgeIsOpen = false
let fridgeContent = ["milk", "eggs", "leftovers"]

func fridgeContains(_ food: String) -> Bool {
	fridgeIsOpen = true
	defer {
		fridgeIsOpen = false
	}

	let result = fridgeContent.contains(food)
	return result
}
fridgeContains("banana")
print(fridgeIsOpen)
```
