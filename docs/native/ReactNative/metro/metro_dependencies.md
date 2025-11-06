---
title: metro依赖解析流程
sidebar_label: metro依赖解析流程
date: 2024-09-23
last_update:
  date: 2024-09-23
---

# metro依赖解析流程

[依赖解析流程](https://metaatem.cn/react/ReactNative%E4%BE%9D%E8%B5%96%E8%A7%A3%E6%9E%90%E6%B5%81%E7%A8%8B.html#%E4%B8%80%E3%80%81%E5%89%8D%E8%A8%80)

## 介绍

此部分主要介绍如何用babel ast将一个模块(也就是一个文件)进行解析并找到其依赖。

treeshaking也是主要在这块进行一些代码定制化。
