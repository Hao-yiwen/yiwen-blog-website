---
title: c中基本数据类型
sidebar_label: c中基本数据类型
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# c中基本数据类型

在 C 语言中，有几种基本数据类型可以用来存储不同类型的数据。Objective-C 的 NSNumber 类可以封装这些基本数据类型，使其可以作为对象使用。

## C 语言的基本数据类型

以下是 C 语言的基本数据类型：

1. 整数类型：

    - int：基本的整数类型。
    - short：短整数类型，通常比 int 短。
    - long：长整数类型，通常比 int 长。
    - long long：超长整数类型。
    - unsigned 类型：对应上述所有类型的无符号版本（如 unsigned int、unsigned short、unsigned long、unsigned long long）。
    - char：字符类型，通常用于表示字符。
    - signed char：有符号字符类型。
    - unsigned char：无符号字符类型。

2. 浮点数类型：

    - float：单精度浮点数类型。
    - double：双精度浮点数类型。
    - long double：扩展精度浮点数类型。

3. 布尔类型（在 C99 及以上版本中引入）：
    - \_Bool：布尔类型。
    - bool：使用 stdbool.h 头文件定义。

## NSNumber 封装 C 的基本数据类型

Objective-C 的 NSNumber 可以封装上述 C 的基本数据类型，使它们可以作为对象使用。以下是 NSNumber 封装不同类型的方法：

1. 整数类型封装：
    - int：numberWithInt:
    - short：numberWithShort:
    - long：numberWithLong:
    - long long：numberWithLongLong:
    - unsigned int：numberWithUnsignedInt:
    - unsigned short：numberWithUnsignedShort:
    - unsigned long：numberWithUnsignedLong:
    - unsigned long long：numberWithUnsignedLongLong:
    - char：numberWithChar:
    - unsigned char：numberWithUnsignedChar:
    - signed char：numberWithChar:（可以使用同一个方法）
2. 浮点数类型封装：
    - float：numberWithFloat:
    - double：numberWithDouble:
    - long double：没有直接的方法，可以使用 double 封装
3. 布尔类型封装：
    - BOOL（Objective-C 定义的布尔类型，实质上是 signed char）：numberWithBool:

## 特定的基础类型和封装类型

1.	int 和 NSInteger
	-	int：标准的 C 语言整数类型，通常是 32 位。
	-	NSInteger：一个整数类型，根据平台的不同，其实际类型可能是 int 或 long。在 32 位平台上，NSInteger 是 int，在 64 位平台上是 long。
```objectivec
int intValue = 42;
NSInteger nsIntegerValue = 42;
```

2.	float 和 CGFloat
	-	float：标准的 C 语言单精度浮点数类型。
	-	CGFloat：一个浮点数类型，根据平台的不同，其实际类型可能是 float 或 double。在 32 位平台上，CGFloat 是 float，在 64 位平台上是 double。
```objectivec
float floatValue = 3.14f;
CGFloat cgFloatValue = 3.14f;  // 在 32 位平台上
CGFloat cgFloatValue64 = 3.14; // 在 64 位平台上
```