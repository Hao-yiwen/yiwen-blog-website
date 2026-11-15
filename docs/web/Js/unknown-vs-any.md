---
title: TypeScript 中 unknown 与 any 的区别
sidebar_label: unknown 与 any 区别
sidebar_position: 4
---

# TypeScript 中 `unknown` 与 `any` 的区别

## 1. 类型安全性

- **`unknown`**：允许将任意类型的值赋给 `unknown` 变量；但在读取属性、调用方法或将其赋值给其他具体类型之前，**必须先进行类型收窄（Type Narrowing）或显式类型断言**。
- **`any`**：完全关闭 TypeScript 对该变量的类型检查，允许任意赋值与任意属性访问，容易将潜在错误遗留到运行时。

## 2. 代码对比

```ts title="使用 any（无编译期保护）"
function logLength(value: any) {
    console.log(value.length); // 编译期不报错，传入数字时在运行时崩溃
}

logLength('Hello, World!'); // 13
logLength([1, 2, 3, 4, 5]); // 5
logLength(42); // undefined / 运行时隐患
```

```ts title="使用 unknown（强制类型收窄）"
function logLength(value: unknown) {
    if (typeof value === 'string' || Array.isArray(value)) {
        console.log(value.length); // 已收窄类型，安全访问
    } else {
        console.log("The value doesn't have a length property");
    }
}
```

## 3. 交叉类型与联合类型行为

- **交叉类型（`&`）**：
  - `unknown & string` $\rightarrow$ `string`（`unknown` 被更具体的类型吸收）
  - `any & string` $\rightarrow$ `any`（`any` 传染整个交叉结果）
- **联合类型（`|`）**：
  - `unknown | string` $\rightarrow$ `unknown`
  - `any | string` $\rightarrow$ `any`

## 总结

`unknown` 是类型安全的顶级类型（Top Type）。面对外部 API 响应、`JSON.parse` 结果或异常捕获对象时，应始终优先使用 `unknown` 替代 `any`。
