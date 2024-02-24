# kotlin中的构造函数

在 Kotlin 中，构造函数用于初始化新创建的对象。Kotlin 处理构造函数的方式稍有不同于其他语言，提供了更简洁的语法以及主构造函数和次构造函数的概念。

## 主构造函数（Primary Constructor）

- 主构造函数是类头的一部分，跟在类名后面。
- 它用于初始化类，并可以直接声明和初始化属性。
- 主构造函数不能包含任何代码。初始化的代码可以放在 init 初始化块中。
```kotlin
class Person(val name: String, var age: Int) {
    init {
        println("Initialized with name: $name and age: $age")
    }
}
```
## 次构造函数（Secondary Constructors）

- 类可以有一个或多个次构造函数。
- 次构造函数是类体内部使用 constructor 关键字定义的。
- 每个次构造函数需要直接或间接地通过另一个构造函数委托给主构造函数，确保主构造函数的初始化逻辑被执行。

```kotlin
class Person(val name: String) {
    init {
        println("First init block: $name")
    }
    var age: Int = 0
    var city: String = "shanghai"


    constructor(name: String, age: Int) : this(name) {
        this.age = age
        println("First secondary constructor: $age")
    }

    constructor(name: String, age: Int, city: String) : this(name) {
        this.age = age
        this.city = city
        println("Second secondary constructor: $city")
    }
    init {
        println("Second init block: $city")
    }
}
fun main(){
    val xiaoming = Person("xiaoming")
    val lili = Person("lili", 18)
    val hanmeimei = Person("hanmeimei", 20, "beijing")
}
```

:::danger
- Kotlin 中的构造函数委托机制是一种确保类的初始化逻辑一致执行的强大工具，通过强制每个次构造函数都委托给主构造函数，它帮助维护了代码的清晰度和稳定性。
- 执行顺序：
    1. 主构造函数
    2. 根据顺序决定执行init块还是次构造函数
:::