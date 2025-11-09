---
title: Python 魔法方法完全指南
sidebar_label: Python 魔法方法
date: 2025-11-09
last_update:
  date: 2025-11-09
---

# Python 魔法方法完全指南

魔法方法（Magic Methods）也被称为双下划线方法（Dunder Methods），是 Python 中以双下划线 `__` 开头和结尾的特殊方法。它们让你能够自定义对象的行为，使类的实例能够像内置类型一样工作。

## `__call__` - 让对象可调用

`__call__` 是最有趣的魔法方法之一，它让对象实例可以像函数一样被调用。

### 基本用法

```python
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x + self.n

# 创建对象
add_5 = Adder(5)

# 像函数一样调用对象！
result = add_5(10)  # 调用 __call__(10)
print(result)  # 15

# 检查是否可调用
print(callable(add_5))  # True
```

### 为什么使用 `__call__`？

使用 `__call__` 的主要优势：

1. **有状态的函数**：对象可以保存状态，而普通函数需要使用全局变量或闭包
2. **更清晰的接口**：当需要配置参数时，构造函数提供了清晰的接口
3. **面向对象设计**：可以利用继承和多态
4. **框架集成**：许多框架（如 PyTorch）使用这种模式

## `__call__` 的实际应用

### 1. 装饰器

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call #{self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("Alice")  # Call #1, Hello, Alice!
say_hello("Bob")    # Call #2, Hello, Bob!
print(f"Total calls: {say_hello.count}")  # Total calls: 2
```

### 2. 有状态的函数

```python
class LinearModel:
    """一个简单的线性模型"""
    def __init__(self, weight=0.5, bias=0.1):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        """预测函数"""
        return self.weight * x + self.bias

    def update(self, new_weight, new_bias):
        """更新模型参数"""
        self.weight = new_weight
        self.bias = new_bias

# 使用
model = LinearModel()
print(model(2))  # 1.1

# 更新参数后
model.update(2, 0)
print(model(2))  # 4.0
```

### 3. PyTorch 中的 `nn.Module`

在 PyTorch 中，所有神经网络模块都可以像函数一样调用，这就是通过 `__call__` 实现的：

```python
import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        """定义前向传播"""
        return self.fc(x)

    # nn.Module 已经定义了 __call__
    # 它会调用 forward() 并处理钩子函数

# 使用
net = MyNetwork()
x = torch.randn(2, 10)

# 这两种方式都可以，但推荐第一种
y = net(x)          # 调用 __call__，会触发 forward() 和钩子函数
# y = net.forward(x)  # 直接调用 forward()，不会触发钩子函数
```

:::tip
在 PyTorch 中，始终使用 `net(x)` 而不是 `net.forward(x)`，因为 `__call__` 会处理训练/评估模式切换和钩子函数。
:::

### 4. 缓存/记忆化

```python
class Memoize:
    """缓存函数结果的装饰器"""
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
            print(f"计算 {args}")
        else:
            print(f"使用缓存 {args}")
        return self.cache[args]

@Memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试
print(fibonacci(5))  # 会看到缓存的效果
print(fibonacci(5))  # 第二次直接使用缓存
```

### 5. 工厂模式

```python
class ShapeFactory:
    """形状工厂类"""
    def __init__(self):
        self._shapes = {}

    def register(self, name, shape_class):
        """注册新的形状类型"""
        self._shapes[name] = shape_class

    def __call__(self, name, *args, **kwargs):
        """创建形状实例"""
        if name not in self._shapes:
            raise ValueError(f"Unknown shape: {name}")
        return self._shapes[name](*args, **kwargs)

# 使用
class Circle:
    def __init__(self, radius):
        self.radius = radius

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

factory = ShapeFactory()
factory.register('circle', Circle)
factory.register('rectangle', Rectangle)

# 像函数一样使用工厂
circle = factory('circle', 5)
rect = factory('rectangle', 10, 20)
```

## 常见魔法方法概览

### 构造与表示

```python
class MyClass:
    def __init__(self, value):
        """构造函数 - 初始化对象"""
        self.value = value

    def __new__(cls, *args, **kwargs):
        """创建实例（很少需要重写）"""
        return super().__new__(cls)

    def __del__(self):
        """析构函数 - 对象被销毁时调用"""
        print(f"Deleting {self.value}")

    def __repr__(self):
        """开发者友好的表示 - repr(obj)"""
        return f"MyClass(value={self.value})"

    def __str__(self):
        """用户友好的表示 - str(obj) 或 print(obj)"""
        return f"MyClass with value: {self.value}"

    def __format__(self, format_spec):
        """格式化字符串 - format(obj, spec)"""
        return f"{self.value:{format_spec}}"
```

### 比较运算符

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        """等于 =="""
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        """不等于 !="""
        return not self.__eq__(other)

    def __lt__(self, other):
        """小于 <"""
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)

    def __le__(self, other):
        """小于等于 <="""
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        """大于 >"""
        return not self.__le__(other)

    def __ge__(self, other):
        """大于等于 >="""
        return not self.__lt__(other)

# 使用
p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1 == p2)  # False
print(p1 < p2)   # True (距离原点更近)
```

:::tip
Python 3.7+ 推荐使用 `@dataclass` 和 `functools.total_ordering` 来简化比较操作的实现。
:::

### 算术运算符

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """加法 +"""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """减法 -"""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """乘法 *"""
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        """除法 /"""
        return Vector(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar):
        """整除 //"""
        return Vector(self.x // scalar, self.y // scalar)

    def __mod__(self, scalar):
        """取模 %"""
        return Vector(self.x % scalar, self.y % scalar)

    def __pow__(self, power):
        """幂运算 **"""
        return Vector(self.x ** power, self.y ** power)

    def __neg__(self):
        """取负 -x"""
        return Vector(-self.x, -self.y)

    def __abs__(self):
        """绝对值 abs(x)"""
        return (self.x**2 + self.y**2) ** 0.5

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# 使用
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)    # Vector(4, 6)
print(v1 * 2)     # Vector(6, 8)
print(-v1)        # Vector(-3, -4)
print(abs(v1))    # 5.0
```

### 容器方法

```python
class MyList:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        """长度 - len(obj)"""
        return len(self.items)

    def __getitem__(self, index):
        """获取元素 - obj[index]"""
        return self.items[index]

    def __setitem__(self, index, value):
        """设置元素 - obj[index] = value"""
        self.items[index] = value

    def __delitem__(self, index):
        """删除元素 - del obj[index]"""
        del self.items[index]

    def __contains__(self, item):
        """包含检查 - item in obj"""
        return item in self.items

    def __iter__(self):
        """迭代 - for x in obj"""
        return iter(self.items)

    def __reversed__(self):
        """反向迭代 - reversed(obj)"""
        return reversed(self.items)

# 使用
my_list = MyList([1, 2, 3, 4])
print(len(my_list))      # 4
print(my_list[0])        # 1
print(2 in my_list)      # True

for item in my_list:
    print(item)          # 1 2 3 4
```

### 上下文管理器

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        """进入 with 块时调用"""
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用"""
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        # 返回 True 会抑制异常，返回 False 或 None 会传播异常
        return False

# 使用
with FileManager('test.txt', 'w') as f:
    f.write('Hello World')
# 文件会自动关闭
```

:::tip
Python 3.7+ 推荐使用 `contextlib.contextmanager` 装饰器来创建简单的上下文管理器。
:::

### 其他实用方法

```python
class SmartClass:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        """让对象可调用"""
        return self.value + x

    def __bool__(self):
        """布尔值转换 - bool(obj) 或 if obj:"""
        return self.value != 0

    def __hash__(self):
        """哈希值 - hash(obj)，用于字典和集合"""
        return hash(self.value)

    def __sizeof__(self):
        """内存大小 - sys.getsizeof(obj)"""
        return object.__sizeof__(self) + self.value.__sizeof__()

    def __getattr__(self, name):
        """访问不存在的属性时调用"""
        return f"Attribute {name} not found"

    def __setattr__(self, name, value):
        """设置属性时调用"""
        print(f"Setting {name} = {value}")
        super().__setattr__(name, value)

# 使用
obj = SmartClass(10)
print(obj(5))        # __call__: 15
print(bool(obj))     # __bool__: True
print(hash(obj))     # __hash__: 返回哈希值
```

## 完整魔法方法列表

### 构造与销毁
- `__new__(cls, ...)` - 创建实例
- `__init__(self, ...)` - 初始化实例
- `__del__(self)` - 析构函数

### 表示
- `__repr__(self)` - 开发者表示
- `__str__(self)` - 用户表示
- `__format__(self, format_spec)` - 格式化
- `__bytes__(self)` - 字节表示

### 比较运算符
- `__eq__(self, other)` - `==`
- `__ne__(self, other)` - `!=`
- `__lt__(self, other)` - `<`
- `__le__(self, other)` - `<=`
- `__gt__(self, other)` - `>`
- `__ge__(self, other)` - `>=`

### 算术运算符
- `__add__(self, other)` - `+`
- `__sub__(self, other)` - `-`
- `__mul__(self, other)` - `*`
- `__truediv__(self, other)` - `/`
- `__floordiv__(self, other)` - `//`
- `__mod__(self, other)` - `%`
- `__pow__(self, other)` - `**`
- `__matmul__(self, other)` - `@` (矩阵乘法)

### 一元运算符
- `__neg__(self)` - `-x`
- `__pos__(self)` - `+x`
- `__abs__(self)` - `abs(x)`
- `__invert__(self)` - `~x`

### 增强赋值
- `__iadd__(self, other)` - `+=`
- `__isub__(self, other)` - `-=`
- `__imul__(self, other)` - `*=`
- 等等...

### 类型转换
- `__int__(self)` - `int(x)`
- `__float__(self)` - `float(x)`
- `__bool__(self)` - `bool(x)`
- `__complex__(self)` - `complex(x)`

### 容器方法
- `__len__(self)` - `len(x)`
- `__getitem__(self, key)` - `x[key]`
- `__setitem__(self, key, value)` - `x[key] = value`
- `__delitem__(self, key)` - `del x[key]`
- `__contains__(self, item)` - `item in x`
- `__iter__(self)` - `iter(x)`
- `__reversed__(self)` - `reversed(x)`

### 属性访问
- `__getattr__(self, name)` - 访问不存在的属性
- `__setattr__(self, name, value)` - 设置属性
- `__delattr__(self, name)` - 删除属性
- `__getattribute__(self, name)` - 访问任何属性

### 描述符
- `__get__(self, obj, type=None)` - 获取属性值
- `__set__(self, obj, value)` - 设置属性值
- `__delete__(self, obj)` - 删除属性

### 上下文管理
- `__enter__(self)` - 进入 with 块
- `__exit__(self, exc_type, exc_val, exc_tb)` - 退出 with 块

### 可调用对象
- `__call__(self, ...)` - 使对象可调用

### 其他
- `__hash__(self)` - `hash(x)`
- `__dir__(self)` - `dir(x)`
- `__sizeof__(self)` - `sys.getsizeof(x)`

## 实用示例：综合应用

### 智能计数器

```python
class Counter:
    """一个功能丰富的计数器类"""

    def __init__(self, start=0, step=1):
        self.value = start
        self.step = step
        self._history = [start]

    def __call__(self):
        """调用时递增"""
        self.value += self.step
        self._history.append(self.value)
        return self.value

    def __str__(self):
        return f"Counter(value={self.value}, step={self.step})"

    def __repr__(self):
        return f"Counter(start={self._history[0]}, step={self.step})"

    def __int__(self):
        return self.value

    def __add__(self, other):
        """支持加法"""
        return Counter(self.value + other, self.step)

    def __eq__(self, other):
        if isinstance(other, Counter):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, Counter):
            return self.value < other.value
        return self.value < other

    def __len__(self):
        """返回历史记录长度"""
        return len(self._history)

    def __getitem__(self, index):
        """访问历史记录"""
        return self._history[index]

    def __iter__(self):
        """迭代历史记录"""
        return iter(self._history)

    def reset(self):
        """重置计数器"""
        self.value = self._history[0]
        self._history = [self.value]

# 使用示例
counter = Counter(0, 5)
print(counter)        # Counter(value=0, step=5)

counter()             # 递增到 5
counter()             # 递增到 10
print(counter)        # Counter(value=10, step=5)
print(int(counter))   # 10

# 查看历史
print(list(counter))  # [0, 5, 10]
print(counter[1])     # 5

# 比较
print(counter == 10)  # True
print(counter < 15)   # True

# 加法
counter2 = counter + 5
print(counter2)       # Counter(value=15, step=5)
```

### 延迟计算的表达式

```python
class LazyExpression:
    """延迟计算的数学表达式"""

    def __init__(self, value):
        if callable(value):
            self._compute = value
        else:
            self._compute = lambda: value

    def __call__(self):
        """计算并返回结果"""
        return self._compute()

    def __add__(self, other):
        """延迟加法"""
        return LazyExpression(lambda: self() + (other() if isinstance(other, LazyExpression) else other))

    def __mul__(self, other):
        """延迟乘法"""
        return LazyExpression(lambda: self() * (other() if isinstance(other, LazyExpression) else other))

    def __str__(self):
        return f"LazyExpression(result={self()})"

# 使用
x = LazyExpression(10)
y = LazyExpression(20)

# 构建表达式（不立即计算）
expr = (x + y) * LazyExpression(2)

# 只有在调用时才计算
print(expr())  # 60
```

## 检查对象能力

```python
def inspect_object_capabilities(obj):
    """检查对象支持哪些操作"""
    capabilities = {
        'callable': callable(obj),
        'iterable': hasattr(obj, '__iter__'),
        'has_length': hasattr(obj, '__len__'),
        'indexable': hasattr(obj, '__getitem__'),
        'hashable': hasattr(obj, '__hash__'),
        'comparable': hasattr(obj, '__eq__'),
        'context_manager': hasattr(obj, '__enter__') and hasattr(obj, '__exit__'),
    }

    print(f"Capabilities of {type(obj).__name__}:")
    for capability, supported in capabilities.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {capability}")

# 测试
inspect_object_capabilities(Counter())
inspect_object_capabilities([1, 2, 3])
inspect_object_capabilities(lambda x: x)
```

## 最佳实践

### 1. 实现 `__repr__` 时遵循约定

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        # 好的做法：返回可以重建对象的表达式
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        # 用户友好的表示
        return f"({self.x}, {self.y})"

# 理想情况下应该满足：
p = Point(1, 2)
# eval(repr(p)) == p  # 可以通过 repr 重建对象
```

### 2. 比较运算符使用 `@functools.total_ordering`

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor

    def __eq__(self, other):
        return (self.major, self.minor) == (other.major, other.minor)

    def __lt__(self, other):
        return (self.major, self.minor) < (other.major, other.minor)

    # total_ordering 会自动生成其他比较方法

v1 = Version(1, 2)
v2 = Version(1, 3)
print(v1 < v2)   # True
print(v1 <= v2)  # True (自动生成)
print(v1 > v2)   # False (自动生成)
```

### 3. 使用 `__slots__` 优化内存

```python
class Point:
    __slots__ = ['x', 'y']  # 只允许这些属性

    def __init__(self, x, y):
        self.x = x
        self.y = y

# 好处：
# 1. 减少内存使用（没有 __dict__）
# 2. 更快的属性访问
# 3. 防止添加新属性
```

### 4. 实现容器时保持一致性

```python
class Container:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, item):
        return item in self.items

    # 如果实现了 __getitem__，也应该考虑实现 __setitem__ 和 __delitem__
```

## 总结

### 何时使用魔法方法？

1. **让类的行为像内置类型**：实现容器、数字类型等
2. **提供直观的接口**：使用运算符而不是方法名
3. **框架集成**：许多框架依赖特定的魔法方法
4. **性能优化**：某些魔法方法可以提供更高效的实现

### 常用魔法方法速查

```python
class Example:
    # 必备
    __init__       # 构造
    __repr__       # 表示
    __str__        # 字符串

    # 容器
    __len__        # 长度
    __getitem__    # 索引访问
    __iter__       # 迭代
    __contains__   # in 运算符

    # 比较
    __eq__         # ==
    __lt__         # <

    # 运算
    __add__        # +
    __mul__        # *

    # 特殊
    __call__       # 可调用
    __enter__/__exit__  # 上下文管理
    __hash__       # 哈希
```

### 核心要点

1. `__call__` 让对象可调用，常用于有状态的函数和框架集成
2. `__repr__` 应该返回可以重建对象的字符串
3. `__str__` 应该返回用户友好的字符串
4. 实现运算符时保持数学一致性
5. 使用 `@total_ordering` 简化比较运算符
6. 谨慎使用 `__getattr__` 和 `__setattr__`，避免无限递归
7. 上下文管理器用 `contextlib.contextmanager` 更简单

魔法方法让 Python 类可以无缝集成到语言的其余部分，使代码更加 Pythonic 和直观！🎯
