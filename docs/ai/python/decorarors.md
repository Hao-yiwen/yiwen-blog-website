---
title: Python 装饰器简介
sidebar_label: Python 装饰器简介
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# Python 装饰器简介

## 什么是装饰器？

装饰器（Decorator）是 Python 中一种特殊的语法，用于在不修改原函数代码的情况下，为函数添加额外的功能。装饰器本质上是一个函数，它接收一个函数作为参数，并返回一个新的函数。

## 基本语法

使用 `@` 符号将装饰器应用到函数上：

```python
@decorator
def function():
    pass
```

这等价于：

```python
def function():
    pass
function = decorator(function)
```

## 简单示例

### 1. 基础装饰器

```python
def my_decorator(func):
    def wrapper():
        print("函数执行前")
        func()
        print("函数执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

输出：
```
函数执行前
Hello!
函数执行后
```

### 2. 带参数的装饰器

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"函数返回: {result}")
        return result
    return wrapper

@my_decorator
def add(a, b):
    return a + b

add(3, 5)
```

输出：
```
调用函数: add
函数返回: 8
```

### 3. 实用示例：计时装饰器

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()
```

### 4. 带参数的装饰器

```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"你好, {name}!")

greet("小明")
```

输出：
```
你好, 小明!
你好, 小明!
你好, 小明!
```

## 常见应用场景

1. **日志记录** - 记录函数的调用信息
2. **性能测试** - 测量函数执行时间
3. **权限验证** - 检查用户权限
4. **缓存** - 缓存函数结果
5. **重试机制** - 自动重试失败的函数
6. **参数验证** - 验证函数参数

## 内置装饰器

Python 提供了一些常用的内置装饰器：

- `@staticmethod` - 静态方法
- `@classmethod` - 类方法
- `@property` - 将方法转换为属性
- `@functools.wraps` - 保留原函数的元信息

## 最佳实践

使用 `functools.wraps` 保留原函数的元信息：

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

## 总结

装饰器是 Python 中强大且优雅的特性，它遵循开闭原则（对扩展开放，对修改关闭），让代码更加简洁和可维护。掌握装饰器能够帮助你写出更加 Pythonic 的代码。