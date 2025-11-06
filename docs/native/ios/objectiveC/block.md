---
title: OC中的块
sidebar_label: OC中的块
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# OC中的块

在 Objective-C 中，块（Blocks）是一种在C、C++和Objective-C中添加的语法特性，用于定义和传递内联代码段。块类似于其他编程语言中的lambda表达式或匿名函数。它们可以捕获和存储局部变量，并在稍后执行。块是一等公民，可以作为参数传递给方法或函数，作为返回值，甚至可以存储在集合中。

## 块的定义和使用

### 定义一个块

块的基本语法如下：
```objectivec
returnType (^blockName)(parameterTypes) = ^returnType(parameters) {
    // Block body
};
```

### 示例

```objectivec
// 定义一个不带参数和返回值的块
void (^simpleBlock)(void) = ^{
    NSLog(@"This is a simple block");
};

// 调用块
simpleBlock();

// 定义一个带参数和返回值的块
int (^sumBlock)(int, int) = ^int(int a, int b) {
    return a + b;
};

// 调用块
int result = sumBlock(3, 4);
NSLog(@"Sum: %d", result);
```