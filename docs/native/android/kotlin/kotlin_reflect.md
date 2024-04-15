# ::操作符

Java中无法将一个函数作为参数传递给函数或者赋值给其他变量，但是在kotlin中这是可能的，这一切依赖于反射操作符。

## 介绍

在 Kotlin 中，:: 是一个反射操作符，用于引用函数或属性而不是调用它。当你在函数名前面加上 ::，你实际上是在创建该函数的一个引用，这允许你将函数作为参数传递给其他函数，或者将其赋值给变量等。这种机制称为函数引用（Function Reference）。

## 示例

```kotlin
fun tick() {
    println("tick test")
}

fun executeFunction(action: () -> Unit) {
    action()
}

fun main() {
    executeFunction(::tick) // 输出: tick test
}
```