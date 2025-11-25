---
title: Medium
date: 2024-04-12
---

import rn_old from '@site/static/img/rn_old.webp'
import rn_new from '@site/static/img/rn_new.webp'

# Medium

## 介绍

Medium是一个在线发布平台，于2012年由Twitter的共同创始人Evan Williams创建。它被设计为一个简洁的博客平台，旨在鼓励用户撰写和分享各种长度的文章。

<!-- truncate -->

## 背景

2024年4月12日晚上，我在看一篇介绍rn新架构和旧架构的文章，我刚开始觉得写的还行，后面越看越吃惊，因为个人感觉真的写的让人通俗易懂，很多耳熟能详但是很少有文章写到这种程度，于是我选择medium作为我的一个阅读板块，并抽时间进行学习和阅读。

## 入坑文章

-   [rn架构演化](https://medium.com/@under_the_hook/react-native-the-new-architecture-c4ba8ed8b452)

<img src={rn_old} width={800} />

-   旧架构
    -   旧架构依赖于bridge和json异步消息队列的方式传递数据
    -   因为是异步消息队列，所以js端和native互相不可知，导致维护两颗虚拟树，有巨大内存损耗
    -   bridge是单线程，导致如果异步消息队列中数据量过大时候会白屏
    -   因为是异步消息队列，在list中滑动会有白屏现象，因为数据异步同步
    -   因为是异步消息队列，所以native侧在app启动时候需要加载全部模块，因为native和js完全不透明

<img src={rn_new} width={800} />
- 新架构
    - 使用c++的jsi作为抽象层，模拟web浏览器中js直接调用c++底层逻辑
    - js和native都为互操作层，数据实时传输
    - 无需维护两颗虚拟树并异步同步，减少内存损耗
    - 因为js和native直接调用操作，所以应用可以懒加载所需模块
    - jsi提供了接口规范和类型安全，提高代码质量和减少了app崩溃率
