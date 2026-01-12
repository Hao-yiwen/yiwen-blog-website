---
title: Go 语言排序算法面试指南
sidebar_label: 排序算法面试
sidebar_position: 4
tags: [go, 算法, 排序, 面试, 快速排序, 归并排序, 堆排序]
---

# Go 语言排序算法面试指南

Go 语言语法简洁，切片 (Slice) 操作方便，非常适合在面试中手写算法。以下是面试中最常考的 5 种排序算法的 **Go 语言标准实现**。

## 第一梯队：必考算法 (The Big Three)

这三个算法是面试的重中之重，请务必烂熟于心。

### A. 快速排序 (Quick Sort)

这是最经典的写法，使用 `partition` 函数进行分区。

- **时间复杂度：** 平均 $O(n \log n)$，最坏 $O(n^2)$
- **空间复杂度：** $O(\log n)$（递归栈）
- **稳定性：** 不稳定

```go
package main

// QuickSort 入口
func QuickSort(arr []int) {
	quickSortRec(arr, 0, len(arr)-1)
}

// 递归函数
func quickSortRec(arr []int, low, high int) {
	if low < high {
		pivotIndex := partition(arr, low, high)
		quickSortRec(arr, low, pivotIndex-1)  // 排左边
		quickSortRec(arr, pivotIndex+1, high) // 排右边
	}
}

// 分区操作 (核心逻辑)
func partition(arr []int, low, high int) int {
	pivot := arr[high] // 选最后一个元素作为基准
	i := low - 1       // i 指向小于基准的区域边界

	for j := low; j < high; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i] // 交换
		}
	}
	// 将基准元素放到正确的位置 (i+1)
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}
```

### B. 归并排序 (Merge Sort)

Go 的切片特性让归并排序写起来非常优雅。

- **时间复杂度：** $O(n \log n)$（稳定）
- **空间复杂度：** $O(n)$（需要额外数组）
- **稳定性：** 稳定

```go
func MergeSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}

	mid := len(arr) / 2
	// 递归切割：注意 Go 的切片是左闭右开 [0:mid)
	left := MergeSort(arr[:mid])
	right := MergeSort(arr[mid:])

	return merge(left, right)
}

// 合并两个有序切片
func merge(left, right []int) []int {
	result := make([]int, 0, len(left)+len(right))
	i, j := 0, 0

	for i < len(left) && j < len(right) {
		if left[i] <= right[j] { // <= 保证稳定性
			result = append(result, left[i])
			i++
		} else {
			result = append(result, right[j])
			j++
		}
	}

	// 将剩余元素追加到结果中
	result = append(result, left[i:]...)
	result = append(result, right[j:]...)
	return result
}
```

### C. 堆排序 (Heap Sort)

堆排序的核心是 `heapify`（下沉）操作。注意 Go 中一般手动实现堆来展示对原理的理解，而不是直接调 `container/heap` 包。

- **时间复杂度：** $O(n \log n)$（稳定）
- **空间复杂度：** $O(1)$（原地排序）
- **稳定性：** 不稳定

```go
func HeapSort(arr []int) {
	n := len(arr)

	// 1. 建堆：从最后一个非叶子节点开始，自下而上 heapify
	for i := n/2 - 1; i >= 0; i-- {
		heapify(arr, n, i)
	}

	// 2. 排序：交换堆顶元素与末尾元素，并缩小堆范围
	for i := n - 1; i > 0; i-- {
		arr[0], arr[i] = arr[i], arr[0] // 最大的移到末尾
		heapify(arr, i, 0)              // 对剩余元素重新堆化
	}
}

// 下沉操作：维护大顶堆性质
func heapify(arr []int, n int, i int) {
	largest := i
	left := 2*i + 1
	right := 2*i + 2

	// 比较左孩子
	if left < n && arr[left] > arr[largest] {
		largest = left
	}
	// 比较右孩子
	if right < n && arr[right] > arr[largest] {
		largest = right
	}

	// 如果父节点不是最大的，交换并递归
	if largest != i {
		arr[i], arr[largest] = arr[largest], arr[i]
		heapify(arr, n, largest)
	}
}
```

---

## 第二梯队：基础算法

这两个算法代码简单，但要注意 Go 的一些写法细节。

### D. 冒泡排序 (Bubble Sort)

加了 `swapped` 标志位进行优化，这是面试官想看到的细节。

- **时间复杂度：** 平均 $O(n^2)$，最好 $O(n)$
- **空间复杂度：** $O(1)$
- **稳定性：** 稳定

```go
func BubbleSort(arr []int) {
	n := len(arr)
	for i := 0; i < n-1; i++ {
		swapped := false
		// 注意：内层循环只需要走到 n-1-i
		for j := 0; j < n-1-i; j++ {
			if arr[j] > arr[j+1] {
				arr[j], arr[j+1] = arr[j+1], arr[j]
				swapped = true
			}
		}
		// 如果这一轮没有交换，说明已经有序，提前退出
		if !swapped {
			break
		}
	}
}
```

### E. 插入排序 (Insertion Sort)

适合小规模或基本有序的数据。

- **时间复杂度：** 平均 $O(n^2)$，最好 $O(n)$
- **空间复杂度：** $O(1)$
- **稳定性：** 稳定

```go
func InsertionSort(arr []int) {
	n := len(arr)
	for i := 1; i < n; i++ {
		key := arr[i]
		j := i - 1
		// 将比 key 大的元素向后移
		for j >= 0 && arr[j] > key {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = key
	}
}
```

---

## 复杂度速查表

| 算法 | 平均时间 | 最坏时间 | 空间 | 稳定性 |
|------|----------|----------|------|--------|
| **快速排序** | $O(n \log n)$ | $O(n^2)$ | $O(\log n)$ | 不稳定 |
| **归并排序** | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ | 稳定 |
| **堆排序** | $O(n \log n)$ | $O(n \log n)$ | $O(1)$ | 不稳定 |
| **冒泡排序** | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 |
| **插入排序** | $O(n^2)$ | $O(n^2)$ | $O(1)$ | 稳定 |

---

## Go 语言面试特别提示

### 1. 切片 (Slice) 的引用传递

Go 的切片本质上是一个结构体（包含指针、长度、容量）。当你把 `[]int` 传给函数时，修改底层数组的内容（如 `arr[i] = x`）**会**影响原数组。

所以上面的 `QuickSort`, `HeapSort`, `BubbleSort`, `InsertionSort` 都是**原地排序** (In-place)，不需要返回新的切片。

详见：[引用类型 vs 值类型速查表](./reference-vs-value-types.md)

### 2. 内置 `sort` 包

面试官可能会问："Go 标准库的 `sort.Ints()` 用的是什么算法？"

**答案：** Go 的标准库使用了 **pdqsort** (Pattern-Defeating Quicksort)，这是一种混合排序算法，结合了快速排序、堆排序和插入排序的优点，性能非常强劲（Go 1.19+ 版本）。

```go
import "sort"

func main() {
    arr := []int{3, 1, 4, 1, 5, 9, 2, 6}
    sort.Ints(arr) // 使用 pdqsort
    // arr 现在是 [1, 1, 2, 3, 4, 5, 6, 9]
}
```

### 3. 自定义排序

```go
// 按绝对值排序
sort.Slice(arr, func(i, j int) bool {
    return abs(arr[i]) < abs(arr[j])
})

// 结构体排序
type Person struct {
    Name string
    Age  int
}

people := []Person{{"Alice", 30}, {"Bob", 25}}
sort.Slice(people, func(i, j int) bool {
    return people[i].Age < people[j].Age
})
```

---

## 相关阅读

- [引用类型 vs 值类型速查表](./reference-vs-value-types.md) - 理解 slice 传参
- [核心算法套路详解](/docs/cs_base/arithmetic/core_algorithms) - 动态规划、DFS、BFS 等
