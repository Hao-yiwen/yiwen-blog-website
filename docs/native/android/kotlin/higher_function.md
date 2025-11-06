---
title: kotlin常用的高阶函数
sidebar_label: kotlin常用的高阶函数
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# kotlin常用的高阶函数

## forEach()
- 用途：对集合的每个元素应用给定的函数，不返回任何结果

## map()
- 用途：对集合的每个元素应用给定的函数，并返回结果列表。

## filter()
- 用途：返回符合给定条件的所有元素的列表。

## groupBy()
```kt
val groupByMenu = cookies.groupBy {
    it.softBaked
}
val softBakedMenu = groupByMenu[true] ?: listOf()
val crunchyMenu = groupByMenu[false] ?: listOf()
```

## fold()
```kt
// 计算总价格，类似于reduce
val sum = listOf(1, 2, 3).fold(0) { sum, element -> sum + element }
```

## sortedBy()
