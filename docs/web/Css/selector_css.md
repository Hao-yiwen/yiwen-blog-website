---
sidebar_position: 2
---

# css选择器

## 选择器优先级

1. 内联样式：直接写在 HTML 元素中的样式具有最高优先级。
2. ID 选择器（如 #someId）：优先级次之。
3. 类选择器（如 .someClass）、属性选择器（如 [type="text"]）和伪类（如 :hove，:first-child）：优先级再次之。
4. 标签选择器（如 h1、div）和伪元素（如 ::before）：优先级最低。

```css
.markdown > h1,
.markdown h1:first-child {
    --ifm-h1-font-size: 30px;
}
```

在这个例子中：

-   .markdown > h1 和 .markdown h1:first-child 都有一个类选择器（.markdown）和一个标签选择器（h1）。
-   .markdown h1:first-child 还有一个伪类选择器（:first-child）。
-   所以，.markdown h1:first-child 的优先级实际上是更高的，因为它有一个额外的伪类选择器。

:::tip
.parent > div 选择 .parent 的所有直接 div 子元素。
.parent div 选择 .parent 下的所有 div 子元素，不论它们是直接子元素还是更深层次的嵌套元素。
:::
