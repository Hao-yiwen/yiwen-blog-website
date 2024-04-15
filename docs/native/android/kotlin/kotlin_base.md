# Kotlin基础

## 介绍

Kotlin是一种现代的静态类型编程语言，由JetBrains公司开发并在2011年首次公开发布。它被设计为完全兼容Java，同时引入了更简洁的语法和新的特性，以提高开发效率和程序的可读性。

## 语法

-   Kotlin中的Unit相当于ts中的void

### for
```kotlin
<!-- 通过索引遍历 -->
for (index in items.indices) {
    println("item at $index is ${items[index]}")
}
<!-- 遍历集合 -->
val items = listOf("apple", "banana", "kiwifruit")
for (item in items) {
    println(item)
}
```
### when
```
fun describe(obj: Any): String =
    when (obj) {
        1          -> "One"
        "Hello"    -> "Greeting"
        is Long    -> "Long"
        !is String -> "Not a string"
        else       -> "Unknown"
    }
//sampleEnd

fun main() {
    println(describe(1))
    println(describe("Hello"))
    println(describe(1000L))
    println(describe(2))
    println(describe("other"))
}
```

### 继承

### 简单继承

```kotlin
open class Parent {
    fun parentMethod() {
        println("This is a parent method.")
    }
}

class Child : Parent() {
    fun childMethod() {
        println("This is a child method.")
    }
}
```

### 构造函数继承
```kotlin
open class Parent(message: String) {
    init {
        println(message)
    }
}

class Child(message: String) : Parent(message)
```

### 方法重写
```kotlin
open class Parent {
    open fun greet() {
        println("Hello from Parent")
    }
}

class Child : Parent() {
    override fun greet() {
        println("Hello from Child")
    }
}
```

### 集合
```kotlin
<!-- 有序集合 -->
val items = listOf("apple", "banana","dsad")
<!-- 无序集合 -->
val items = setOf("apple", "banana","dsad")
```
- setOf 创建的是一个Set集合，其中的元素是唯一的，不允许重复。Set代表了一个无序集合，主要用于当你关心某个元素是否出现在集合中，而不关心它出现的顺序或次数时。

- listOf 创建的是一个List集合，允许包含重复的元素，并且元素是有序的，即元素的添加顺序被保留。List适用于当元素的顺序重要，或者需要存储重复元素时。