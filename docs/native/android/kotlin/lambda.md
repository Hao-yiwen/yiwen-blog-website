# Kotlin中的Lambda表达式

## 基本语法

```kt
val sum: (Int, Int) -> Int = { a, b -> a + b }
```

sum 是一个变量，引用了一个接收两个 Int 类型参数并返回一个 Int 类型结果的函数。Lambda 表达式由花括号包围，参数列表在 -> 符号之前，函数体在 -> 符号之后。

## 省略规则

-   在 Kotlin 中，如果编译器能够从上下文中推断出 Lambda 表达式的参数类型，则在 Lambda 表达式内部不需要显式声明参数类型。

```kt
val printer: (String) -> Unit = { text -> println(text) }
```

-   **单参数 Lambda：如果 Lambda 表达式只有一个参数，可以不声明参数名，使用默认名 it**

```kt
val square: (Int) -> Int={
    it * it
}
```

-   返回值：Lambda 表达式中的最后一个表达式被视为返回值。

## 使用示例

```kt
<!-- 函数中使用lambda -->
fun operate(x: Int, y: Int, operation: (Int, Int) -> Int): Int {
    return operation(x, y)
}

fun main() {
    val result = operate(2, 3){
        x,y->x+y
    }
    println(result) // 输出：5
}
<!-- forEach使用lambda -->
var sum = 0
val numbers = listOf(1, 2, 3)
numbers.forEach { sum += it }
println(sum) // 输出：6
```
