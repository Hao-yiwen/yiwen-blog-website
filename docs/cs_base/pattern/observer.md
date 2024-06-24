# 观察者模式

## 代码

[具体示例](https://github.com/Hao-yiwen/android-study/tree/master/DesignPatterns/src/main/java/org/example/observer)

## 概述

观察者模式是一种行为设计模式，它定义了一种一对多的依赖关系，当一个对象的状态发生变化时，所有依赖于它的对象都会得到通知并自动更新。观察者模式主要用于实现事件处理系统。

## 结构

观察者模式主要包含以下几个部分：

1.	主题 (Subject)：持有对观察者对象的引用，提供注册和移除观察者对象的方法。
2.	观察者 (Observer)：定义一个更新接口，以便在主题状态发生变化时得到通知。
3.	具体主题 (Concrete Subject)：实现主题接口，维护其状态，当状态发生变化时通知所有观察者。
4.	具体观察者 (Concrete Observer)：实现观察者接口，以便在得到通知时更新自身状态。