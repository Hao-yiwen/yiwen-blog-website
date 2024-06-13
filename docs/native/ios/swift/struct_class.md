# struct和class区别

## 值类型 vs. 引用类型：

-   Struct（结构体）：是值类型。当你创建一个结构体的实例并将其赋值给一个变量或常量，或者传递给一个函数时，实际上是创建了这个实例的一个副本。修改副本不会影响原始实例。
-   Class（类）：是引用类型。当你创建一个类的实例并将其赋值给一个变量或常量，或者传递给一个函数时，实际上是传递了对这个实例的引用。修改引用会影响到原始实例。

## 继承：

-   Struct：不支持继承。你不能从一个结构体继承另一个结构体。
-   Class：支持继承。你可以创建一个类，继承另一个类的属性和方法。

## 构造器：

-   Struct：自动生成一个成员逐一构造器（memberwise initializer），你可以用它来初始化结构体实例。
-   Class：没有自动生成的成员逐一构造器，你需要手动定义构造器（initializer）。

## 类型转换：

-   Struct：不支持类型转换。
-   Class：支持类型转换，可以用 is 和 as 进行类型检查和转换。

## 析构器：

-   Struct：没有析构器（deinitializer）。
-   Class：可以定义析构器，在实例释放前执行一些清理工作。

## 可变性：

-   Struct：如果结构体实例被声明为 let（常量），则所有属性都是不可变的。要想让某个属性可变，必须将该属性声明为 var 并且结构体实例本身也必须是 var。
-   Class：即使类的实例被声明为 let（常量），只要它的属性被声明为 var，这些属性依然可以被修改。

## 示例
```swift
struct MyStruct {
    var name: String
    var age: Int
}

class MyClass {
    var name: String
    var age: Int
    
    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}

// 使用 struct
var struct1 = MyStruct(name: "Alice", age: 25)
var struct2 = struct1
struct2.name = "Bob"
print(struct1.name) // 输出 "Alice"
print(struct2.name) // 输出 "Bob"

// 使用 class
var class1 = MyClass(name: "Alice", age: 25)
var class2 = class1
class2.name = "Bob"
print(class1.name) // 输出 "Bob"
print(class2.name) // 输出 "Bob"
```