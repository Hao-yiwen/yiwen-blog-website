---
title: OC常见类型
sidebar_label: OC常见类型
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# OC常见类型

## 基础类型扩展

除了 C 语言的基本数据类型（如 int、float、double、char 等），Objective-C 还提供了一些新的基本类型和对象类型。

- NSNumber
- NSInteger
- CGFloat

:::danger
CGFloat 和 NSInteger 都不是对象，它们是基础数据类型的别名，设计用于在不同平台上提供一致的行为和性能。
:::

## 对象类型

1. NSObject

NSObject 是大多数 Objective-C 类的基类，提供了许多基础的方法和属性。大多数自定义类都会继承自 NSObject。

```objectivec
@interface MyClass : NSObject
// ...
@end
```

2. NSString

NSString 是不可变的字符串类，用于表示文本。

```objectivec
NSString *string = @"Hello, World!";
```

3. NSMutableString

NSMutableString 是可变的字符串类，用于需要修改字符串内容的情况。
```objectivec
NSMutableString *mutableString = [NSMutableString stringWithString:@"Hello"];
[mutableString appendString:@", World!"];
```

4. NSArray

NSArray 是不可变的数组类，用于存储有序的对象集合。
```objetivec
NSArray *array = @[@"Apple", @"Banana", @"Cherry"];
```

5. NSMutableArray

NSMutableArray 是可变的数组类，允许在数组中添加或移除元素。
```objectivec
NSMutableArray *mutableArray = [NSMutableArray arrayWithArray:@[@"Apple", @"Banana"]];
[mutableArray addObject:@"Cherry"];
```

6. NSDictionary

NSDictionary 是不可变的字典类，用于存储键值对。
```objectivec
NSDictionary *dictionary = @{@"name": @"John", @"age": @30};
```

7. NSMutableDictionary

NSMutableDictionary 是可变的字典类，允许在字典中添加或移除键值对。
```objectivec
NSMutableDictionary *mutableDictionary = [NSMutableDictionary dictionaryWithDictionary:@{@"name": @"John"}];
[mutableDictionary setObject:@30 forKey:@"age"];
```
8. NSNumber

NSNumber 是用于封装基本数据类型（如 int、float 等）的对象类型。

```objectivec
NSNumber *number = @42;
NSNumber *floatNumber = @3.14;
```

9. NSValue

NSValue 是用于封装非对象数据类型（如结构体、指针等）的对象类型。

```
NSRect rect = NSMakeRect(0, 0, 100, 100);
NSValue *rectValue = [NSValue valueWithRect:rect];
```

## 常用类

1. NSDate

NSDate 是用于表示日期和时间的类。
```objectivec
NSDate *date = [NSDate date];
```

2. NSData

NSData 是用于表示任意二进制数据的类。
```objectivec
NSData *data = [NSData dataWithContentsOfFile:@"path/to/file"];
```

3. NSSet

NSSet 是用于存储唯一对象的无序集合。

```objetivec
NSSet *set = [NSSet setWithObjects:@"Apple", @"Banana", @"Cherry", nil];
```

4. NSMutableSet

NSMutableSet 是可变的无序集合，允许添加和移除对象。
```objetivec
NSMutableSet *mutableSet = [NSMutableSet setWithObjects:@"Apple", @"Banana", nil];
[mutableSet addObject:@"Cherry"];
```

## 动态类型

1. id

id 是一种动态类型，可以指向任何 Objective-C 对象。
```objetivec
id obj = @"Hello, World!";
```