---
sidebar_position: 1
---

# Objective C

## 介绍

Objective-C（OC），作为一种基于 C 的编程语言，继承了 C 语言的基本类型，并引入了一些自己的特定类型。

## 文档

-   [24天学习oc](https://www.binpress.com/learn-objective-c-24-days/)

-   [oc参考文档](https://www.tutorialspoint.com/objective_c/objective_c_structures.htm)

## 数据类型

### 基础数据类型（从 C 继承）

#### 整型（Integer Types）

-   int: 标准整型，通常是 32 位。
-   short: 短整型，通常是 16 位。
-   long: 长整型，32 位或 64 位，取决于系统。
-   long long: 更长的整型，通常是 64 位。

#### 浮点类型（Floating-Point Types）

-   float: 单精度浮点类型。
-   double: 双精度浮点类型。

#### 字符类型（Character Types）

-   char: 用于表示单个字符，通常是 8 位。

#### 布尔类型（Boolean Type）

-   BOOL: Objective-C 中的布尔类型，其值通常是 YES 或 NO。

### 对象类型

-   NSObject
    Objective-C 中大多数类的基类。
-   NSString
    用于表示不可变字符串。
-   NSMutableString
    用于表示可变字符串。
-   NSArray
    用于表示不可变数组。
-   NSMutableArray
    用于表示可变数组。
-   NSDictionary
    用于表示不可变键值对集合。
-   NSMutableDictionary
    用于表示可变键值对集合。
-   NSNumber
    用于封装基本数值为对象。
-   NSDate
    用于表示日期和时间。

### 特殊类型

-   id
    一个通用类型的对象指针。
-   SEL
    用于表示方法选择器。
-   Class
    用于表示类对象。
-   Block
    用于表示代码块。
-   void
    用于表示无返回值。

### 指针类型

指针类型用于存储内存地址，常见于对象类型和 C 风格字符串。

### C语言结构体和联合体

-   struct: 结构体，用于存储不同类型的数据。
-   union: 联合体，多个成员共享同一块内存。

### 枚举类型

-   enum: 用于定义一组命名的整型常量。

## 为什么基本数据类型在oc开发中并不常见

虽然 Objective-C 支持 C 语言的基本类型，但在实际的 iOS 和 macOS 开发中，更倾向于使用 NS 类型，因为它们提供了更高级的功能，更好的集成了 Objective-C 的面向对象特性，以及与 Apple 的框架和生态系统的兼容性。基本数据类型通常用于更底层的操作或与 C 代码的互操作性。
