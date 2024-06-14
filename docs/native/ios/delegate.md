# 委托模式

委托是一种设计模式，允许一个对象将其某些操作委托给另一个对象处理，委托方法是由委托对象实现的方法，当特定事件发生的时候，由被委托对象调用这些方法。

## 作用

委托模式在 iOS 中广泛使用，主要原因是它可以实现对象之间的松耦合。通过视图控制器传递委托，可以确保委托关系在正确的上下文中被设置和管理，避免全局变量带来的不必要的复杂性和潜在问题。这样可以使代码更模块化、更易维护，同时也更符合面向对象设计的原则。

## 示例

[代码示例](https://github.com/Hao-yiwen/ios-study/blob/master/ios-study/views/IOSBaseScreen/delegateScreen/TopContainerController.swift)
