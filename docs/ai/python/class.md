---
title: Python 类（Class）快速入门
sidebar_label: Python 类（Class）快速入门
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# Python 类（Class）快速入门

## 什么是类？

**类（Class）** 是对象的模板，**对象（Object）** 是类的实例。

```python
# 类比：类=汽车设计图，对象=具体的汽车
class Dog:
    pass

my_dog = Dog()      # 创建对象
your_dog = Dog()    # 创建另一个对象
```

---

## 基本语法

```python
class Dog:
    def __init__(self, name, age):
        """初始化方法，创建对象时自动调用"""
        self.name = name    # 实例属性
        self.age = age
    
    def bark(self):
        """实例方法"""
        print(f"{self.name}在汪汪叫")

# 创建对象
dog = Dog("旺财", 3)
dog.bark()  # 旺财在汪汪叫
```

### 关键要点

- **`__init__`**: 构造函数，初始化对象属性
- **`self`**: 代表对象本身，必须是方法的第一个参数
- **实例属性**: `self.name`，每个对象独有
- **实例方法**: 操作对象的函数

---

## self 的含义

`self` 就是对象本身，让不同对象能访问自己的属性。

```python
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        print(f"{self.name}在叫")  # 通过 self 访问自己的属性

dog1 = Dog("旺财")
dog2 = Dog("小黑")

dog1.bark()  # 旺财在叫 ← self 指向 dog1
dog2.bark()  # 小黑在叫 ← self 指向 dog2
```

---

## 继承

子类可以继承父类的属性和方法。

```python
# 父类
class Animal:
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        print(f"{self.name}在吃东西")

# 子类
class Dog(Animal):  # 继承 Animal
    def bark(self):
        print(f"{self.name}在汪汪叫")

# 使用
dog = Dog("旺财")
dog.eat()   # 继承的方法：旺财在吃东西
dog.bark()  # 自己的方法：旺财在汪汪叫
```

### super() 调用父类

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类的 __init__
        self.breed = breed

dog = Dog("旺财", "金毛")
print(f"{dog.name}是{dog.breed}")  # 旺财是金毛
```

**为什么需要 `super().__init__()`？**
- 初始化父类 `nn.Module`
- 让 PyTorch 能追踪模型参数
- 启用 `.parameters()`, `.cuda()`, `.train()` 等功能

---

## 常用特殊方法

特殊方法以双下划线 `__` 开头和结尾。

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        """定义 print() 输出"""
        return f"Point({self.x}, {self.y})"
    
    def __add__(self, other):
        """定义 + 运算符"""
        return Point(self.x + other.x, self.y + other.y)
    
    def __eq__(self, other):
        """定义 == 运算符"""
        return self.x == other.x and self.y == other.y

p1 = Point(1, 2)
p2 = Point(3, 4)

print(p1)           # Point(1, 2)
p3 = p1 + p2        # 调用 __add__
print(p3)           # Point(4, 6)
print(p1 == p2)     # False
```

### 常用特殊方法

| 方法 | 说明 | 触发方式 |
|------|------|----------|
| `__init__` | 初始化 | 创建对象时 |
| `__str__` | 字符串表示 | `print(obj)` |
| `__len__` | 长度 | `len(obj)` |
| `__getitem__` | 索引访问 | `obj[i]` |
| `__add__` | 加法 | `obj1 + obj2` |
| `__eq__` | 相等 | `obj1 == obj2` |

---

## 类属性 vs 实例属性

```python
class Dog:
    species = "犬科"  # 类属性，所有实例共享
    
    def __init__(self, name):
        self.name = name  # 实例属性，每个实例独有

dog1 = Dog("旺财")
dog2 = Dog("小黑")

print(Dog.species)    # 犬科
print(dog1.species)   # 犬科
print(dog1.name)      # 旺财
print(dog2.name)      # 小黑
```

---

## 实战例子

### 银行账户

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        print(f"存款 {amount}，余额 {self.balance}")
    
    def withdraw(self, amount):
        if amount > self.balance:
            print("余额不足！")
        else:
            self.balance -= amount
            print(f"取款 {amount}，余额 {self.balance}")

account = BankAccount("张三", 1000)
account.deposit(500)    # 存款 500，余额 1500
account.withdraw(300)   # 取款 300，余额 1200
```

### PyTorch 风格的神经网络

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # ← 必须调用！
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet(10, 20, 2)
print(model)
```

**为什么需要 `super().__init__()`？**
```python
# ✅ 正确
class Net(nn.Module):
    def __init__(self):
        super().__init__()  # 必须先调用
        self.fc = nn.Linear(10, 5)

# ❌ 错误
class BrokenNet(nn.Module):
    def __init__(self):
        # 没有调用 super().__init__()
        self.fc = nn.Linear(10, 5)  # 参数无法被追踪！
```

---

## 类方法和静态方法

```python
class MyClass:
    count = 0
    
    def __init__(self):
        MyClass.count += 1
    
    @classmethod
    def get_count(cls):
        """类方法，访问类属性"""
        return cls.count
    
    @staticmethod
    def add(x, y):
        """静态方法，不需要 self 或 cls"""
        return x + y

obj1 = MyClass()
obj2 = MyClass()

print(MyClass.get_count())  # 2
print(MyClass.add(3, 5))    # 8
```

---

## 总结

### 核心要点

```python
class ClassName(ParentClass):
    # 类属性
    class_var = "共享"
    
    # 初始化
    def __init__(self, param):
        self.instance_var = param  # 实例属性
    
    # 实例方法
    def method(self):
        return self.instance_var
    
    # 特殊方法
    def __str__(self):
        return f"ClassName({self.instance_var})"
```

### 记住这些

1. **`__init__`** - 构造函数，初始化对象
2. **`self`** - 代表对象本身
3. **继承** - `class Child(Parent)`
4. **`super()`** - 调用父类方法
5. **特殊方法** - `__str__`, `__add__` 等自定义行为

### PyTorch 中的类

```python
# 标准模板
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # ← 第一件事！
        # 定义层
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 2)
    
    def forward(self, x):
        # 定义前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```