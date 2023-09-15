---
sidebar_position: 3
---

# 各类css

## SCSS (Sassy CSS)

### 由来

SCSS 是 Sass 3 的一种新语法，全称是 Sassy CSS。

### 语法

更接近于 CSS，有大括号和分号。

### 特性:

-   变量

```scss
$font-stack: Helvetica, sans-serif;
$primary-color: #333;

body {
    font: 100% $font-stack;
    color: $primary-color;
}
```

-   嵌套

```scss
nav {
    ul {
        margin: 0;
        padding: 0;
        list-style: none;
    }

    li {
        display: inline-block;
    }

    a {
        display: block;
        padding: 6px 12px;
        text-decoration: none;
    }
}
```

-   Mixins

```scss
// transform
@mixin transform($property) {
    -webkit-transform: $property;
    -ms-transform: $property;
    transform: $property;
}

.box {
    @include transform(rotate(30deg));
}

// radius
@mixin border-radius($radius) {
    -webkit-border-radius: $radius;
    -moz-border-radius: $radius;
    -ms-border-radius: $radius;
    border-radius: $radius;
}
.button {
    @include border-radius(3px);
}
```

-   函数

```scss
@function set-text-color($type) {
    @if $type == light {
        @return #333;
    } @else {
        @return #fff;
    }
}

body {
    color: set-text-color(light);
}
```

-   继承

```scss
.message {
    border: 1px solid #ccc;
    padding: 10px;
    color: #333;
}

.success {
    @extend .message;
    border-color: green;
}
```

## Sass (Syntactically Awesome StyleSheets)

### 由来

Sass 是一种成熟、稳定、强大的 CSS 扩展语言，由 Hampton Catlin 设计。Sass 是一个 CSS 预处理器，它允许你使用变量、嵌套规则、函数和混合等功能，以便编写更易维护和组织的 CSS 代码。

### 语法

没有大括号{}和分号;。

### 特性

和scss相比，只是没有大括号{}和分号;。

## Less (Leaner Style Sheets)

### 由来

Less 由 Alexis Sellier 设计，目标是为 CSS 添加一些有用的扩展。

### 语法

类似于 CSS。

### 特性:

-   变量

```less
@font-stack: 'Helvetica, sans-serif';
@primary-color: #333;

body {
    font-family: @font-stack;
    color: @primary-color;
}
```

-   Mixins

```less
.border-radius(@radius) {
    -webkit-border-radius: @radius;
    -moz-border-radius: @radius;
    border-radius: @radius;
}

.button {
    .border-radius(3px);
}
```

-   嵌套

```less
nav {
    ul {
        list-style: none;
    }
    li {
        display: inline-block;
    }
}
```

-   函数

```less
@base: 5%;
@filler: @base * 2;

.progress {
    width: @base + @filler;
}
```

-   继承

```less
.message {
    border: 1px solid #ccc;
    padding: 10px;
    color: #333;
}

.success:extend(.message) {
    border-color: green;
}
```

## less和scss区别

1. 语法

Less: 使用 @ 前缀定义变量。

```less
@primary-color: #333;
```

SCSS: 使用 $ 前缀定义变量。

```scss
$primary-color: #333;
```

2. 语言特性

-   Less: 提供了 JavaScript 运行时环境，可以在浏览器中运行（尽管通常还是在构建过程中编译）。
-   SCSS: 主要是 Ruby-based，但现在有一个非常流行的 C/C++ 版本（LibSass）。通常需要预编译。

3. 拓展能力

-   Less: 内置功能较少，但可以通过 JavaScript 插件进行拓展。

-   SCSS: 内置功能更多，包括颜色函数、列表函数、以及对字符串、数字等更多的操作。

4. 继承

-   Less: 使用 :extend 进行继承。

-   SCSS: 使用 @extend 进行继承。

5. 循环和控制指令

-   Less: 有基础的条件和循环语句，但比 SCSS 更为简单。

-   SCSS: 提供了更丰富的控制指令，包括 @if，@for，@each 和 @while

6. 社群和生态系统

-   Less: 在过去几年的发展中相对平稳，但似乎没有 SCSS 那么活跃。

-   SCSS: 更广泛地被接受和使用，有更大的社群和更多的贡献。

## scss颜色函数

1. 调整亮度
   使用 lighten() 和 darken() 函数可以调整颜色的亮度。

```scss
$primary-color: #337ab7;

.light-button {
    background-color: lighten($primary-color, 20%); // 变亮 20%
}

.dark-button {
    background-color: darken($primary-color, 20%); // 变暗 20%
}
```

2. 调整透明度
   使用 rgba() 和 fade-out() 函数来调整颜色的透明度。

```scss
$primary-color: #337ab7;

.transparent-button {
    background-color: rgba($primary-color, 0.5); // 设置透明度为 0.5
}

.fade-button {
    background-color: fade-out($primary-color, 0.5); // 降低 50% 的透明度
}
```

3. 调整饱和度
   使用 saturate() 和 desaturate() 函数来调整颜色的饱和度。

```scss
$primary-color: #337ab7;

.saturated-button {
    background-color: saturate($primary-color, 20%); // 增加 20% 的饱和度
}

.desaturated-button {
    background-color: desaturate($primary-color, 20%); // 减少 20% 的饱和度
}
```

4. 调整色相
   使用 adjust-hue() 函数可以旋转颜色的色相。

```scss
$primary-color: #337ab7;

.adjusted-button {
    background-color: adjust-hue($primary-color, 60deg); // 色相旋转 60 度
}
```

5. 合并颜色
   使用 mix() 函数可以合并两种颜色。

```scss
$color1: #337ab7;
$color2: #4caf50;

.mixed-button {
    background-color: mix($color1, $color2, 50%); // 合并两种颜色，权重为 50%
}
```
