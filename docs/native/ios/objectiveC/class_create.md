---
title: 为什么oc中类创建不同于c
sidebar_label: 为什么oc中类创建不同于c
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 为什么oc中类创建不同于c

- c
```
SampleClass *sampleClass = (SampleClass *)malloc(sizeof(SampleClass));
```
1. 这种方式虽然可以分配内存，但在 Objective-C 中不推荐使用，因为它无法正确调用对象的初始化方法。此外，malloc 分配的内存并不与 Objective-C 的内存管理系统兼容。

- oc
```
SampleClass *sampleClass = [[SampleClass alloc] init];
```
1. alloc：分配内存。它为对象分配足够的内存，并返回一个指向该内存的指针。
2. init：初始化对象。在内存分配之后，init 方法初始化对象的状态。