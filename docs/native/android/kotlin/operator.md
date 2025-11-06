---
title: kotlin中的特殊操作符
sidebar_label: kotlin中的特殊操作符
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# kotlin中的特殊操作符

Kotlin 中确实有一些特殊的符号，它们在特定的上下文中具有独特的用途。除了之前提到的展开操作符 \*，这里列出一些其他特殊符号及其用途：

## 展开操作符 \*

```kt
fun sum(vararg numbers:Int, test:String){
    println(numbers.sum())
}

sum(test = "test",numbers = *intArrayOf(1,2,3,4))
```

用于在函数调用时展开数组或可变参数（vararg）。

## 安全调用操作符 ?.

允许在一个对象可能为 null 的情况下安全地调用其方法或访问其属性。如果对象为 null，则不执行调用并返回 null。

## Elvis 操作符 ?:

当表达式左侧的值不为 null 时返回左侧的值，否则返回右侧的值。常用于提供 null 值的默认值。

## 非空断言操作符 !!

用于将任何值转换为非空类型，如果值为 null 则抛出一个异常。

## Lambda 表达式箭头 ->

在 Lambda 表达式中分隔参数列表和Lambda体。

## 集合过滤操作符 in/!in

检查元素是否在集合中或不在集合中。

## 类型检查操作符 is/!is

检查对象是否是特定类型或不是特定类型。

## 引用操作符 ::

用于引用类成员或函数，而不是调用它。可以用于方法引用或属性引用。

```kt
fun isOdd(x: Int) = x % 2 != 0
val predicate: (Int) -> Boolean = ::isOdd
println(predicate(5)) // 输出：true
```

## 区间操作符 ..

创建一个区间，常用于循环中。

## 解构声明

使用 (a, b) = pair 来解构一个对象。虽然不是操作符，但是解构声明使用特殊的语法结构，使得从对象中提取多个属性变得简单。
