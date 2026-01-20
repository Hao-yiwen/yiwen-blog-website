---
title: Python @dataclass 终极指南
sidebar_label: Dataclass 装饰器
tags: [python, dataclass, 装饰器]
---

# Python `@dataclass` 终极指南

## 1. 什么是 `@dataclass`？

`@dataclass` 是 Python 3.7 引入的一个非常强大的装饰器，旨在**通过自动生成代码来减少样板代码（Boilerplate Code）**。如果你经常编写主要用于存储数据的类，它将彻底改变你的编码习惯。

在 Python 中，我们经常定义一些主要用于保存数据的类。在没有 `@dataclass` 之前，我们需要手动编写 `__init__`、`__repr__`、`__eq__` 等魔法方法。

`@dataclass` 位于 `dataclasses` 模块中，它是一个类装饰器，能够检查类中定义的**类型注解（Type Hints）**，并自动为你生成这些常用方法。

### 核心优势

* **代码简洁**：减少重复的 `self.x = x` 写法。
* **可读性强**：类结构一目了然，像是一张数据表。
* **自带功能**：自动支持打印（repr）、比较（eq）等功能。

---

## 2. 直观对比：传统类 vs Dataclass

让我们通过一个对比来看看它有多省事。

### 传统写法 (Old Way)

你需要手动写很多重复代码：

```python
class User:
    def __init__(self, name: str, age: int, is_active: bool = True):
        self.name = name
        self.age = age
        self.is_active = is_active

    def __repr__(self):
        return f"User(name={self.name!r}, age={self.age}, is_active={self.is_active})"

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return (self.name, self.age, self.is_active) == (other.name, other.age, other.is_active)
        return NotImplemented

# 初始化
user1 = User("Alice", 30)
print(user1)  # User(name='Alice', age=30, is_active=True)
```

### @dataclass 写法 (New Way)

同样的功能，代码量减少 70%：

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int
    is_active: bool = True  # 支持默认值

# 初始化
user1 = User("Alice", 30)
user2 = User("Alice", 30)

print(user1)          # 自动生成的 __repr__: User(name='Alice', age=30, is_active=True)
print(user1 == user2) # 自动生成的 __eq__: True
```

---

## 3. 进阶用法详解

虽然基础用法很简单，但掌握以下进阶技巧才能避免常见的坑。

### 3.1 处理可变默认值 (`field` 和 `default_factory`)

**这是新手最容易踩的坑。**

在 Python 中，千万不要直接将列表（List）或字典（Dict）作为默认参数（例如 `tags: list = []`），这会导致所有实例共享同一个列表。

在 Dataclass 中，你需要使用 `field` 和 `default_factory`：

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Team:
    name: str
    # 错误做法: members: List[str] = []
    # 正确做法: 使用 default_factory
    members: List[str] = field(default_factory=list)

t1 = Team("Alpha")
t1.members.append("Bob")

t2 = Team("Beta")
print(t2.members) # 输出 []，是安全的，不会包含 "Bob"
```

### 3.2 不可变数据对象 (`frozen=True`)

如果你希望创建的对象在初始化后**不能被修改**（类似元组，或者是为了作为字典的 key），可以使用 `frozen=True`。

```python
@dataclass(frozen=True)
class Config:
    host: str
    port: int

conf = Config("localhost", 8080)
# conf.port = 9090  # 抛出 FrozenInstanceError 异常
```

### 3.3 初始化后的处理 (`__post_init__`)

因为 `__init__` 是自动生成的，如果你需要在初始化后执行某些逻辑（比如验证数据、根据其他字段计算新字段），可以使用 `__post_init__`。

```python
@dataclass
class Rect:
    width: float
    height: float
    area: float = field(init=False) # init=False 表示不需要在初始化时传参

    def __post_init__(self):
        # 在 __init__ 执行完后自动调用
        self.area = self.width * self.height

r = Rect(2, 5)
print(r.area) # 10.0
```

### 3.4 转换为字典或元组

Dataclasses 内置了辅助函数，方便将对象转换为 JSON 友好的格式。

```python
from dataclasses import asdict, astuple

@dataclass
class Point:
    x: int
    y: int

p = Point(10, 20)

print(asdict(p))  # {'x': 10, 'y': 20} -> 非常适合 API 返回
print(astuple(p)) # (10, 20)
```

---

## 4. 参数配置速查表

`@dataclass` 装饰器本身接受多个参数来控制生成的代码行为：

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| `init` | `True` | 自动生成 `__init__` 方法 |
| `repr` | `True` | 自动生成 `__repr__` (打印字符串) |
| `eq` | `True` | 自动生成 `__eq__` (等于比较) |
| `order` | `False` | 自动生成 `<` `>` `<=` `>=` 方法 (用于排序) |
| `unsafe_hash` | `False` | 强制生成 `__hash__` (通常配合 `frozen=True` 使用) |
| `frozen` | `False` | 禁止修改属性 (变为不可变对象) |

**示例：支持排序**

```python
@dataclass(order=True)
class Player:
    score: int
    name: str

p1 = Player(90, "A")
p2 = Player(85, "B")

print(p1 > p2) # True (先比 score，若相同则比 name)
```

---

## 5. 总结

`@dataclass` 是现代 Python (3.7+) 开发中定义数据结构的首选方式。

* **什么时候用？** 当你的类主要是为了存储数据（DTOs, 配置对象, 数据库模型映射）。
* **什么时候不用？** 当你的类行为非常复杂，需要非常定制化的 `__init__` 逻辑，或者需要兼容非常老的 Python 版本时。
