---
sidebar_position: 2
---

# Swift基础总结

## 泛型比较

### JAVA

```java
// 泛型类
public class Box<T> {
    private T t;

    public void set(T t) { this.t = t; }
    public T get() { return t; }
}

// 使用
Box<Integer> integerBox = new Box<Integer>();
integerBox.set(new Integer(10));
Integer intValue = integerBox.get();

// 泛型方法
public static <U> void display(U element){
    System.out.println(element);
}
```

### swift

```swift
// 泛型函数
func swapTwoValues<T>(_ a: inout T, _ b: inout T) {
    let temporaryA = a
    a = b
    b = temporaryA
}

// 使用
var someInt = 3
var anotherInt = 107
swapTwoValues(&someInt, &anotherInt)

// 泛型类型
struct Stack<Element> {
    var items = [Element]()
    mutating func push(_ item: Element) {
        items.append(item)
    }
    mutating func pop() -> Element {
        return items.removeLast()
    }
}
```

### JS

```ts
// 泛型函数
function identity<T>(arg: T): T {
    return arg;
}

// 使用
let output = identity<string>('myString');

// 泛型类
class GenericNumber<T> {
    zeroValue: T;
    add: (x: T, y: T) => T;
}

let myGenericNumber = new GenericNumber<number>();
myGenericNumber.zeroValue = 0;
myGenericNumber.add = function (x, y) {
    return x + y;
};
```

## &操作符

在 Swift 中，& 符号用于表示一个变量的引用，或者更具体地说，表示一个变量的地址。当你在函数调用中使用 & 符号时，你是在传递一个指向该变量的引用，而不是变量的值。

这在 Swift 中是用于实现所谓的 "in-out 参数"。当你有一个函数参数标记为 inout 时，它意味着你可以在函数内部修改该参数，且修改会影响到原始变量。

```swift
func swapTwoValues<T>(_ a: inout T, _ b: inout T) {
    let temporaryA = a
    a = b
    b = temporaryA
}

var someInt = 3
var anotherInt = 107
swapTwoValues(&someInt, &anotherInt)
print(someInt) // 107
print(anotherInt)  // 3
```

由于使用了 & 符号，函数内部的变化会影响到 someInt 和 anotherInt，因此在函数调用后，someInt 的值将为 107，而 anotherInt 的值将为 3。

## 结构体中mutating关键字

```swift
struct Stack<Element> {
    var items = [Element]()
    mutating func push(_ item: Element) {
        items.append(item)
    }
    mutating func pop() -> Element {
        return items.removeLast()
    }
}
```

在 Swift 中，结构体（struct）是值类型。这意味着当你将一个结构体赋值给一个变量、常量或者传递给一个函数时，它实际上是被复制的。因此，结构体的属性默认是不可修改的，即使它们被定义为变量。

mutating 关键字用于结构体和枚举中的方法，它表示这个方法将会修改结构体或枚举的实例本身或其任何属性。在方法体内，self 引用代表了该结构体的一个可变版本。

在你给出的 Stack 结构体示例中，push 和 pop 方法都可能会修改 items 数组，因此它们都需要被标记为 mutating。

当你使用这个结构体时，只有当它被存储在一个变量中时，你才能调用其 mutating 方法。如果它被存储在一个常量中，那么你不能调用这些方法，因为常量的内容是不可变的。

```swift
var myStack = Stack<Int>()
myStack.push(5)   // 这是合法的，因为 myStack 是一个变量

let constantStack = Stack<Int>()
// constantStack.push(5)   // 这将会报错，因为 constantStack 是一个常量
```

## 范围操作符

`..<`：半开范围操作符（Half-open Range Operator）

例如，`0..<4` 创建一个范围，从 0 开始到 3 结束，但不包括 4。所以，当你在循环中使用 `for i in 0..<4` 时，i 的值将是 0, 1, 2, 3。

`...`：封闭范围操作符（Closed Range Operator）

例如，`0...5` 创建一个范围，从 0 开始到 5 结束，并包括 5。所以，当你在循环中使用 `for i in 0...5` 时，i 的值将是 0, 1, 2, 3, 4, 5。

```swift
// ..<操作符
var total = 0
for i in 0..<4 {
    total += i // 6
}
// ...操作符
var total = 0
for i in 0..<4 {
    total += i // 10
}
```
