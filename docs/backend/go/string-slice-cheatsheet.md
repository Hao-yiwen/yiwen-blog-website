---
sidebar_position: 12
title: 字符串与切片操作速查表
tags: [go, string, slice, 切片, 字符串, 面试, 速查表]
---

# Go 字符串与切片操作速查表

本文整理了 Go 语言中字符串和数组/切片的常用操作，偏实战 + 面试高频。

## 一、字符串常用操作

### 1. 获取长度

```go
s := "hello"
len(s) // 字节长度
utf8.RuneCountInString(s) // 字符长度（中文要用这个）
```

### 2. 遍历字符串（推荐方式）

```go
for i, ch := range s {
    fmt.Println(i, string(ch))
}
```

> range 按 rune 遍历，支持中文

### 3. 字符串转数组

```go
arr := []byte(s)       // 按字节
runes := []rune(s)     // 按字符
```

### 4. 数组转字符串

```go
s := string(arr)
s := string(runes)
```

### 5. 拼接字符串

```go
s := "a" + "b"
```

高性能拼接：

```go
var sb strings.Builder
sb.WriteString("a")
sb.WriteString("b")
res := sb.String()
```

### 6. 分割 & 拼接

```go
parts := strings.Split("a,b,c", ",")
s := strings.Join(parts, "-")
```

### 7. 是否包含 / 前后缀

```go
strings.Contains(s, "he")
strings.HasPrefix(s, "he")
strings.HasSuffix(s, "lo")
```

### 8. 替换

```go
strings.Replace(s, "a", "b", -1)
```

### 9. 去空格

```go
strings.TrimSpace(s)
strings.Trim(s, "a")
```

## 二、数组 / 切片常用操作

### 1. 定义

```go
arr := [3]int{1,2,3}
slice := []int{1,2,3}
```

### 2. 添加元素

```go
slice = append(slice, 4)
```

### 3. 删除元素（索引 i）

```go
slice = append(slice[:i], slice[i+1:]...)
```

### 4. 拷贝

```go
b := make([]int, len(a))
copy(b, a)
```

### 5. 切片截取

```go
sub := slice[1:3]
```

### 6. 遍历

```go
for i, v := range slice {
    fmt.Println(i, v)
}
```

### 7. 排序

```go
sort.Ints(slice)
sort.Strings(strSlice)
```

### 8. 反转数组

```go
for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
    slice[i], slice[j] = slice[j], slice[i]
}
```

### 9. 查找

```go
for i, v := range slice {
    if v == target {
        return i
    }
}
```

## 三、字符串 + 数组经典面试写法

### 反转字符串

```go
r := []rune(s)
for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
    r[i], r[j] = r[j], r[i]
}
res := string(r)
```

### 判断回文

```go
l, r := 0, len(runes)-1
for l < r {
    if runes[l] != runes[r] {
        return false
    }
    l++
    r--
}
return true
```

### 双指针数组求和

```go
l, r := 0, len(nums)-1
for l < r {
    sum := nums[l] + nums[r]
    if sum == target {
        return []int{l, r}
    } else if sum < target {
        l++
    } else {
        r--
    }
}
```

## 四、Go 面试最爱考的坑

### 1. 字符串不可修改

```go
s[0] = 'a' // ❌ 编译报错
```

必须转 slice 才能修改。

### 2. len 是字节不是字符

```go
len("中") == 3
```

### 3. append 可能扩容

```go
a := []int{1,2}
b := append(a, 3)
// b 可能是新数组
```

:::tip 面试建议
熟练掌握以上操作，特别是双指针、反转、回文判断，这些是算法面试的高频考点。
:::
