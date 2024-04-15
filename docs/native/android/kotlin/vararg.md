# vararg

在 Kotlin 中，vararg 关键字用于表示一个函数参数可以接受可变数量的参数。这意味着你可以传递零个、一个或多个参数给这个函数，而这些参数将作为数组的形式在函数体内被访问。vararg 的使用使得函数调用更加灵活，无需事先知道将会传递给函数多少个参数，也无需创建一个数组来传递多个参数。

## 使用 vararg 的例子

假设你想要写一个函数，这个函数可以接受任意数量的整数，并计算它们的和。使用 vararg，你可以这样定义这个函数：

```kt
fun sum(vararg numbers: Int): Int {
    return numbers.sum()
}

val result1 = sum(1, 2, 3) // 传递三个参数
val result2 = sum() // 不传递任何参数
val result3 = sum(10, 20) // 传递两个参数
```

## 将 vararg 与其他参数一起使用

如果函数有多个参数，并且 vararg 不是最后一个参数，那么在调用函数时，必须使用命名参数语法来指定 vararg 之后的参数值。如果 vararg 是函数的最后一个参数，这种情况就不会发生。

## 将数组传递给 vararg 参数

如果你已经有一个数组，想要将它传递给一个接受 vararg 参数的函数，你可以使用展开操作符（*）来实现：

```kt
val numbers = intArrayOf(1, 2, 3)
val result = sum(*numbers)
```