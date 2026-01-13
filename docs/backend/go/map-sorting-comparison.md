---
sidebar_position: 13
title: Map 排序与比较
tags: [go, map, 排序, 比较, 面试, 泛型]
---

# Go Map 排序与比较

Go map 面试必考点：map 排序和 map 比较。本文提供标准工程级实现模板。

## 一、Map 排序（按 key 排序）

因为 map 无序，必须转 slice 排。

### 按 key 排序输出

```go
m := map[string]int{
    "c": 3,
    "a": 1,
    "b": 2,
}

keys := make([]string, 0, len(m))
for k := range m {
    keys = append(keys, k)
}

sort.Strings(keys)

for _, k := range keys {
    fmt.Println(k, m[k])
}
```

## 二、按 value 排序

```go
type KV struct {
    Key   string
    Value int
}

list := make([]KV, 0, len(m))
for k, v := range m {
    list = append(list, KV{k, v})
}

sort.Slice(list, func(i, j int) bool {
    return list[i].Value < list[j].Value
})

for _, kv := range list {
    fmt.Println(kv.Key, kv.Value)
}
```

## 三、Map 比较（判断两个 map 是否相等）

### 标准比较函数

```go
func mapEqual(a, b map[string]int) bool {
    if len(a) != len(b) {
        return false
    }

    for k, v := range a {
        if bv, ok := b[k]; !ok || bv != v {
            return false
        }
    }
    return true
}
```

## 四、支持泛型的 Map 比较（Go1.18+）

```go
func mapEqual[K comparable, V comparable](a, b map[K]V) bool {
    if len(a) != len(b) {
        return false
    }
    for k, v := range a {
        if bv, ok := b[k]; !ok || bv != v {
            return false
        }
    }
    return true
}
```

## 五、Map 比较进阶（value 不是可比较类型）

如果 value 是 slice / map：

```go
func mapEqualDeep(a, b map[string][]int) bool {
    if len(a) != len(b) {
        return false
    }
    for k, v := range a {
        bv, ok := b[k]
        if !ok || !reflect.DeepEqual(v, bv) {
            return false
        }
    }
    return true
}
```

## 六、Map 排序 + 比较常见面试问法

### Q1：为什么 map 不能排序？

因为 map 本身无序，只能转 slice 排。

### Q2：为什么 map 不能直接比较？

Go 只允许比较 `nil`，不允许 `==` 比较内容。

## 七、经典 LeetCode 场景

### 统计后排序

```go
cnt := map[string]int{}
cnt["a"]++
cnt["b"]++

type Pair struct {
    K string
    V int
}

arr := []Pair{}
for k, v := range cnt {
    arr = append(arr, Pair{k, v})
}

sort.Slice(arr, func(i, j int) bool {
    return arr[i].V > arr[j].V
})
```

:::tip 面试建议
熟练掌握 map 按 key/value 排序、泛型比较函数，这些是 Go 面试高频考点。
:::
