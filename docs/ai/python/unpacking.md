---
title: 解包
sidebar_label: 解包
date: 2025-11-04
last_update:
  date: 2025-11-05
---

# 解包

**Python 里的 `*` 和 `**` 解包用法详解**

## 基础用法

### 1. 列表（元组）解包：`*`

```python
lst = [1, 2, 3]
a, b, c = lst  # a=1, b=2, c=3

# *用于参数传递
def foo(a, b, c):
    print(a, b, c)

foo(*lst)  # 等价于 foo(1, 2, 3)
```

### 2. 字典解包：`**`

```python
d = {'x': 10, 'y': 20}

def bar(x, y):
    print(x + y)

bar(**d)  # 等价于 bar(x=10, y=20)
```

### 3. 变量收集

```python
a, *b, c = [1, 2, 3, 4, 5]
# a=1, b=[2,3,4], c=5
```

### 4. 合并/拆解容器

```python
lst1 = [1, 2]
lst2 = [3, 4]
merged = [*lst1, *lst2]  # [1, 2, 3, 4]

dict1 = {'a': 1}
dict2 = {'b': 2}
merged_dict = {**dict1, **dict2}  # {'a': 1, 'b': 2}
```

## `*args` vs `**kwargs` 深入对比

### `*args` - 位置参数
接收/传递**按顺序**的参数

```python
def add(a, b, c):
    return a + b + c

# 调用方式1：正常传参
add(1, 2, 3)

# 调用方式2：用 * 解包列表/元组
args = (1, 2, 3)
add(*args)  # 等同于 add(1, 2, 3)
```

### `**kwargs` - 关键字参数
接收/传递**带名字**的参数

```python
def greet(name, age):
    print(f"{name}, {age}岁")

# 调用方式1：正常传参
greet(name="小明", age=18)

# 调用方式2：用 ** 解包字典
kwargs = {"name": "小明", "age": 18}
greet(**kwargs)  # 等同于 greet(name="小明", age=18)
```

## 装饰器中的应用

### 为什么装饰器需要两个？

为了支持**各种调用方式**：

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):  # 接收所有参数
        print("调用前")
        result = func(*args, **kwargs)  # 原样传给原函数
        print("调用后")
        return result
    return wrapper

@my_decorator
def example(a, b, name="默认"):
    print(f"a={a}, b={b}, name={name}")

# 这些调用方式都能工作！
example(1, 2)                    # *args 接收
example(1, 2, name="自定义")      # *args + **kwargs
example(a=1, b=2)                # **kwargs 接收
```

### 直观对比

```python
# 只用 *args - ❌ 不支持关键字参数
def wrapper(*args):
    func(*args)

example(1, 2, name="test")  # 报错！

# 同时用两个 - ✅ 万能
def wrapper(*args, **kwargs):
    func(*args, **kwargs)

example(1, 2, name="test")  # 正常工作
```

## 函数定义中的参数收集

### 接收任意数量的位置参数

```python
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
```

### 接收任意数量的关键字参数

```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="张三", age=25, city="北京")
# name: 张三
# age: 25
# city: 北京
```

### 组合使用

```python
def flexible_func(required, *args, **kwargs):
    print(f"必需参数: {required}")
    print(f"额外位置参数: {args}")
    print(f"关键字参数: {kwargs}")

flexible_func(1, 2, 3, name="test", value=100)
# 必需参数: 1
# 额外位置参数: (2, 3)
# 关键字参数: {'name': 'test', 'value': 100}
```

## 总结

### 简单记忆

- `*` 一个星：**顺序**参数（位置）
- `**` 两个星：**命名**参数（关键字）
- 装饰器用两个：**兼容所有调用方式**

### 常见使用场景

- `*`：序列解包（列表/元组）、接收任意位置参数
- `**`：字典解包、接收任意关键字参数
- 装饰器、函数包装器：`*args, **kwargs` 组合使用
- 容器合并：`[*list1, *list2]`、`{**dict1, **dict2}`
