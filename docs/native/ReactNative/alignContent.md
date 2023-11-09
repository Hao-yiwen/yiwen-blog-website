---
sidebar_position: 4
---

import rn_absolute from "@site/static/img/rn_absoulte.png";

# style

## alignContent和alignItems比较

### 解释

在 React Native 中，alignItems 和 alignContent 是 Flexbox 布局的属性，用于在交叉轴上对齐容器内的子元素。尽管它们听起来相似，但它们在多行布局中的作用是不同的。

### alignItems

-   alignItems 用于在交叉轴上对齐容器内的子项。
-   对于一行（或单行）的Flex容器，alignItems 决定了这一行中子项的对齐方式。
-   默认值通常是 stretch，使得子项拉伸以填满容器在交叉轴上的额外空间。
-   其他值包括 flex-start、flex-end、center 和 baseline。

### alignContent

-   alignContent 用于多行Flex容器，在交叉轴上对齐整个内容。
-   只有当有额外的空间和容器有多行子项时，alignContent 才有效。
-   它决定了多行之间的空间分配和对齐方式。
-   可选值包括 flex-start、flex-end、center、stretch、space-between 和 space-around。

### 说明

-   alignItems 和 alignContent 的默认值都是 stretch。从而多行/列盒子无法紧密连接在一起。
-   alignItems 拥有 flex-start、flex-end、center、stretch 和 baseline属性
-   alignContent 拥有 flex-start、flex-end、center、stretch、space-between 和 space-around属性

### space-between 和 space-around说明

#### space-between

-   子项之间的空间是相等的。
-   第一个子项贴近容器的一端，最后一个子项贴近容器的另一端。
-   不在子项与容器边缘之间添加空间。

示意布局：

```
|[子项]---[子项]---[子项]|
```

#### space-around

-   子项周围的空间是相等的。
-   子项之间的空间是子项与容器边缘之间空间的两倍。
-   每个子项旁边都有等量的空间，包括与容器边缘的空间。

示意布局：

```
|- [子项] -- [子项] -- [子项] -|
```

#### 说明

在 space-around 的情况下，因为子项与容器边缘之间的空间是子项之间空间的一半，所以子项看起来是“浮动”在空间中的，而 space-between 则是将子项均匀分布在容器内，两端的子项紧贴容器边缘。

## flex

在 React Native 中，使用 flex: 1 设置的属性实际上是一个简写，它同时设置了 flex-grow, flex-shrink, 和 flex-basis。具体来说，flex: 1 相当于：

-   flex-grow: 1; —— 表示组件可以伸展以占用多余的空间。
-   flex-shrink: 1; —— 表示组件可以收缩以防止溢出。
-   flex-basis: 0%; —— 通常默认值是 auto，但在 flex: 1 的简写中，默认设置为 0，这意味着组件的初始大小不基于它的内容或者它的宽高属性。

所以，当你在 React Native 中使用 flex: 1 时，你实际上是告诉组件：

-   它可以和兄弟组件一样，根据可用空间进行伸展。
-   在默认情况下，如果父组件的空间不足以容纳所有子组件的基础大小，它也允许这个组件收缩。
-   它的起始大小（flex-basis）是 0，这意味着它的大小完全由 flex 布局的空间分配决定。

## position

在 React Native 中，position 属性用于确定组件如何在父容器中定位。它有两个值：relative 和 absolute。

### relative（默认值）

-   当组件的 position 设置为 relative 时，该组件将根据其正常位置进行定位。此时，您可以使用 top、bottom、left 和 right 属性来调整其位置，这些属性会相对于组件在文档流中的初始位置移动组件。
-   即使您移动了组件，它在布局中占据的空间仍然是基于原始位置的。

### absolute

-   将组件的 position 设置为 absolute 会将组件从正常的文档流中取出，这意味着它不再占据空间，而是相对于其最近的非静态定位的祖先（通常是父容器）进行定位。

<img src={rn_absolute} width={400} />

-   使用 absolute 定位的组件的位置可以通过 top、bottom、left 和 right 属性来指定，这些属性是相对于其父容器的边缘进行定位的。

#### 绝对定位说明

在 React Native 中，如果你将一个子组件设置为绝对定位，那么它会相对于最近的具有相对定位（position: 'relative'）的父组件进行定位。如果没有更近的具有相对定位的父组件，它将相对于它的父容器定位，即默认情况下它会相对于根视图定位
