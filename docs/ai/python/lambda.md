---
title: Lambda 匿名函数
sidebar_label: Lambda 匿名函数
date: 2025-11-05
last_update:
  date: 2025-11-05
---

# Lambda 匿名函数

Python 中的 `lambda` 是**匿名函数**，用于创建简单的一次性函数。

## 基本语法

```python
lambda 参数: 表达式
```

## 常见用法

### 1. 基础示例
```python
# 普通函数
def add(x, y):
    return x + y

# 等价的 lambda
add = lambda x, y: x + y

print(add(3, 5))  # 8
```

### 2. 配合 sorted() 排序
```python
# 按字典的值排序
data = [{'name': 'Alice', 'age': 25},
        {'name': 'Bob', 'age': 20}]

sorted_data = sorted(data, key=lambda x: x['age'])
# 结果：Bob(20), Alice(25)
```

### 3. 配合 map() 处理列表
```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
# [1, 4, 9, 16]
```

### 4. 配合 filter() 过滤
```python
numbers = [1, 2, 3, 4, 5, 6]
even = list(filter(lambda x: x % 2 == 0, numbers))
# [2, 4, 6]
```

### 5. pandas 中使用
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3]})

# 对列应用函数
df['B'] = df['A'].apply(lambda x: x * 2)
```

## 限制

**只能是单个表达式**，不能有多行或复杂逻辑
```python
# 错误 - 不能有多条语句
lambda x:
    result = x * 2
    return result

# 正确 - 使用普通函数
def process(x):
    result = x * 2
    return result
```

## 何时使用

**适合使用**：简单操作、一次性使用、作为参数传递
**不适合使用**：复杂逻辑、需要重用、需要清晰命名

## 更多示例

### 条件表达式
```python
# 使用三元运算符
max_val = lambda x, y: x if x > y else y
print(max_val(10, 5))  # 10
```

### 多个参数
```python
# 计算多个数的和
sum_all = lambda *args: sum(args)
print(sum_all(1, 2, 3, 4))  # 10
```

### 结合 reduce()
```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
# 1 * 2 * 3 * 4 = 24
```

## 最佳实践

1. **保持简单**：如果逻辑复杂，使用普通函数
2. **可读性优先**：如果 lambda 降低可读性，使用 `def`
3. **避免复杂嵌套**：多层 lambda 嵌套会难以理解

```python
# 不推荐 - 难以理解
result = list(map(lambda x: list(map(lambda y: y*2, x)), [[1,2], [3,4]]))

# 推荐 - 清晰明了
def double_nested(nested_list):
    return [[y*2 for y in x] for x in nested_list]

result = double_nested([[1,2], [3,4]])
```
