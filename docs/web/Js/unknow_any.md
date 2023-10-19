---
sidebar_position: 4
---

# unknow和any区别

## 安全性

-   unknown：当你试图将一个类型赋值给 unknown 类型时，TypeScript 不会阻止你。但是，当你试图将 unknown 类型的值赋值给其他类型或尝试调用/构造/访问其成员时，你必须先进行类型检查或类型断言。
-   any：any 类型表示可以赋值给它的任何值，而不会进行类型检查。使用 any 类型可以绕过 TypeScript 的类型检查系统，这可能会导致运行时错误。

## 用例

-   unknown：当你不知道某个值的类型，但仍希望保留 TypeScript 的类型安全性时，应该使用 unknown。这经常出现在动态内容、库的开发或 API 响应中。
-   any：当你确实需要绕过类型检查时使用，例如在迁移到 TypeScript 时或使用老的 JavaScript 库。

```ts title="any"
function logLength(value: any) {
    console.log(value.length); // 这里不会有类型错误，即使某些值可能没有 `.length` 属性
}

logLength('Hello, World!'); // 输出：13
logLength([1, 2, 3, 4, 5]); // 输出：5
logLength(42); // 这里在运行时会抛出错误，因为数字没有 `.length` 属性，但在编译时不会有任何警告或错误
```

```ts title="unknow"
function logLength(value: unknown) {
    if (typeof value === 'string' || Array.isArray(value)) {
        console.log(value.length); // 这里安全，因为我们已经检查了值的类型
    } else {
        console.log("The value doesn't have a length property");
    }
}

logLength('Hello, World!'); // 输出：13
logLength([1, 2, 3, 4, 5]); // 输出：5
logLength(42); // 输出：The value doesn't have a length property
```

## 类型操作

-   unknown：在没有类型断言或基于类型的条件检查的情况下，你不能对 unknown 类型的值进行任何操作。
-   any：你可以对 any 类型的值执行任何操作，而不会得到类型错误。

## 与其他类型的交互

### unknown

-   当与其他类型交叉 (&) 时，结果为其他类型。例如：unknown & string 结果为 string。
-   当与其他类型联合 (|) 时，结果为 unknown。例如：unknown | string 结果为 unknown。

### any

-   当与其他类型交叉 (&) 时，结果为其他类型。例如：any & string 结果为 string。
-   当与其他类型联合 (|) 时，结果为 any。例如：any | string 结果为 any。

## 总结

unknown 和 any 都是 TypeScript 的顶级类型，但 unknown 提供了更好的类型安全性，而 any 则绕过了 TypeScript 的类型系统。在大多数情况下，推荐使用 unknown 而不是 any，以获得更好的类型安全性。
