---
sidebar_position: 1
---

# 介绍

前端web开发

## 学习文档

[javascript高级程序设计第四版](https://yiwen-oss.oss-cn-shanghai.aliyuncs.com/js_develop4.pdf)

[Head First HTML与CSS](https://yiwen-oss.oss-cn-shanghai.aliyuncs.com/head-first-html_css.pdf)

[CSS权威指南](https://yiwen-oss.oss-cn-shanghai.aliyuncs.com/css_3.pdf)

[es6标准入门第三版](https://yiwen-oss.oss-cn-shanghai.aliyuncs.com/es6_3.pdf)

[typescrip学习教程](https://yiwen-oss.oss-cn-shanghai.aliyuncs.com/typescript_study.pdf)

[css资源集合](https://css.doyoe.com/)

## 一些好的Web解决方案和库

-   [bootstrap](https://getbootstrap.com/docs/5.3/forms/form-control/) bootstrap样式库: [bootstrap接入指南](https://github.com/Hao-yiwen/web-study/tree/master/cra-demo)
-   [elementui](https://element.eleme.cn/#/zh-CN) 不用多说
-   [antd](https://ant-design.antgroup.com/index-cn) 不用多说
-   [umi](https://umijs.org/) 一套后台管理系统搭建的解决方案

## qa

### clsx作用

:::note
clsx 的名字可能是由 "class" 和 "extends" 的缩写或组合而来，用于表示这个库的主要用途：扩展和操作 CSS 类名。这个库作为更轻量级和更快速的 classnames 库替代品而生，用于在 JavaScript（特别是在 React 项目中）合并多个类名。
:::

`clsx` 是一个用于在 `JavaScript` 和 `React` 项目中轻松组合 `class` 名称的实用程序库。它是 `classnames` 库的更轻量级和更快速的版本，通常用于条件性地将多个 `class` 名称合并成一个字符串。

这在 React 组件中是非常有用的，特别是当你需要基于组件的 state 或 props 来动态更改 class 名称时。

```jsx
import clsx from 'clsx';

const Button = ({ children, isActive }) => {
    const buttonClass = clsx('button', {
        'is-active': isActive,
        'is-disabled': !isActive,
    });

    return <button className={buttonClass}>{children}</button>;
};
```

`在这个示例中，buttonClass` 将会是 `'button is-active'` 或 `'button is-disabled'`，取决于 `isActive` 的值。
