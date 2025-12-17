---
title: final和static
sidebar_label: final和static
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# final和static

在Java中，final和static是两个用途和含义完全不同的关键字，它们可以用于变量、方法和类。

## final

-   变量：当final关键字用于变量时，这意味着该变量是最终变量，即一旦被初始化之后其值就不能被改变。对于基本数据类型的变量，final确保其数值不变；对于引用类型的变量，final确保引用不会改变，但对象的状态（即对象的属性）可以改变。

-   方法：将方法声明为final意味着该方法不能被子类覆盖（Override）。这常用于锁定方法的实现，防止任何修改其行为的尝试。

-   类：当一个类被声明为final时，表明这个类不能被继承。这对于创建不可变的类或确保安全性（防止类被扩展以改变行为）非常有用。

## static

-   变量：用static声明的变量称为静态变量，它属于类本身而不是类的任何对象实例。所有实例共享同一个静态变量。静态变量在程序开始时创建，在程序结束时销毁。

-   方法：static方法也称为静态方法，它属于类而非类的实例。静态方法可以通过类名直接调用，而不需要创建类的实例。静态方法只能直接访问类的静态成员，不能直接访问非静态成员。

-   块：静态块（static块）用于初始化类的静态变量。静态块在类被加载到JVM时自动执行一次。

## 用途和区别

-   final的用途主要是为了创建不可变的变量、防止继承以及防止方法被覆盖，以此来增加代码的安全性和简洁性。
-   static的用途主要是为了实现类级别的变量和方法，这些成员不依赖于类的实例而存在。静态成员常用于工具方法或常量。

## 示例

```java
public class MyClass {
    public static final int CONSTANT = 10; // 静态常量，属于类且值不可变
    private static int staticVar = 20; // 静态变量，所有实例共享

    public final void finalMethod() {
        // 此方法不能被子类覆盖
    }

    public static void staticMethod() {
        // 静态方法，属于类
    }
}
```
