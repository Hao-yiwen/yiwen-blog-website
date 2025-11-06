---
title: 宏和预处理器
sidebar_label: 宏和预处理器
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 宏和预处理器

## 宏（Macro）

宏是由预处理器处理的文本替换机制。使用宏可以定义常量、函数和代码片段，编译器在编译之前会用宏的定义替换代码中的宏调用。


### 定义宏的语法

```c
#define NAME value
```
- #define：预处理器指令，用于定义宏。
- NAME：宏的名字。
- value：宏的值，可以是常量、表达式或代码片段。

### 常用的宏

1.	常量宏：用于定义常量值。
```
#define p1 3.14159
```

2. 函数宏: 用于定义类似函数的代码片段
```
#define SQUAR(x) ((x) * (x))
```

3. 条件编译宏: 用于控制代码的编译

```c
#define DEBUG
#ifDef DEBUG
#define LOG(msg) printf("DEBUG： %S\n", msg)
#else
#define LOG(msg) printf(msg)
#endif
```

### 宏示例
```
#define MAX_SIZE 100
#define PI 3.14159
#define SQUARE(x) ((x) * (x))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DEBUG
#ifdef DEBUG
#define LOG(msg) printf("DEBUG: %s\n", msg)
#else
#define LOG(msg)
#endif
#ifdef _WIN32
#define PLATFORM "Windows"
#elif __linux__
#define PLATFORM "Linux"
#elif __APPLE__
#define PLATFORM "macOS"
#else
#define PLATFORM "Unknown"
#endif
```

## 预处理器

预处理器是编译器的一部分，他在编译过程中的预处理阶段运行，负责处理所有的预处理命令。预处理器指令通常以#开头，用于定义宏、各类文件、条件编译等。

### 常用预处理器指令

1. #define: 定义宏
```
#define P1 3.1415
```

2. #include: 包含头文件
```
#include<stdio.h>
```

3. #if/#eif/#else/#endif: 条件编译
```
#ifdef DEBUG
#define LOG(msg) printf("DEBUG: %s\n", msg)
#else
#define LOG(msg)
#endif
```

4. #undef: 取消宏定义
```c
#define TEMP 100
#undef TEMP
```

## 示例
```
#include <stdio.h>

// 定义常量宏
#define PI 3.14159
#define MAX_SIZE 100

// 定义函数宏
#define SQUARE(x) ((x) * (x))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// 调试宏
#define DEBUG
#ifdef DEBUG
#define LOG(msg) printf("DEBUG: %s\n", msg)
#else
#define LOG(msg)
#endif

int main() {
    printf("PI: %f\n", PI);
    printf("Square of 5: %d\n", SQUARE(5));
    printf("Min of 3 and 7: %d\n", MIN(3, 7));

    LOG("This is a debug message");

    return 0;
}
```
