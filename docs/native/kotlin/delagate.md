# By关键字

在Kotlin中，by关键字主要用于两个场景：委托属性（Property Delegation）和类委托（Class Delegation）。这里我将提供两个示例，分别展示这两种用法。

## 1. 委托属性（Property Delegation）

Kotlin的委托属性允许你将属性的获取（get）和设置（set）操作委托给另一个对象。这对于将属性的行为委托给框架或库中的通用代码非常有用。

```kt
import kotlin.reflect.KProperty

class Delegate {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): String {
        return "$thisRef, thank you for delegating '${property.name}' to me!"
    }

    operator fun setValue(thisRef: Any?, property: KProperty<*>, value: String) {
        println("$value has been assigned to '${property.name}' in $thisRef.")
    }
}

class Example {
    var p: String by Delegate()
}

fun main() {
    val example = Example()
    println(example.p) // 调用getValue()

    example.p = "New value" // 调用setValue()
}
```

## 2.类委托（Class Delegation）

Kotlin的类委托是一种设计模式的实现，它允许你将一个接口的实现委托给另一个对象。这是一种避免继承的方式，可以使得代码更灵活、更易于维护。

```kt
interface Base {
    val number: Int
    fun print()
}

class BaseImpl(val x: Int) : Base {
    override fun print() { print(x) }
}

class Derived(b: Base) : Base by b

fun main() {
    val b = BaseImpl(10)
    // 将委托的对象传进来 然后复制该对象其他属性和方法
    val b = classDelegate(10)
    var x = Derived(b)
    x.print()
    println(x.number)
}
```

1. 如果我们想让Derived类也实现Base接口，但不想在Derived类中手动实现所有Base接口的方法，我们可以使用委托：
2. 这样，Derived类的所有Base接口方法调用都会被委托给对象b。这里是如何使用它的：
3. 在这个示例中，当我们调用Derived(b).print()时，这个调用被委托给了b对象，即BaseImpl的实例。因此，输出的是10，即BaseImpl中x的值。
