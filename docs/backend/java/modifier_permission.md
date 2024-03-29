# Java中各类修饰符的权限

在Java中，权限修饰符（也称为访问修饰符）定义了类、变量、方法和构造器的访问控制级别。Java提供了四种访问权限修饰符，分别是：private、default（没有指定修饰符时的行为）、protected和public。这些修饰符帮助实现了封装，是面向对象编程的一个核心概念。

1. private

-   作用范围: 只能在定义它们的内部使用
-   使用场景: 用于隐藏类的实现细节和保护类的数据，只能通过类内部的方法访问这些`private`成员

2. default

-   作用范围：只能被同一个包(package)内的类访问
-   使用场景：如果没有指定修饰符（即默认访问权限），那么成员变量，方法，构造器可以被同一个包内任何其他类访问。类在没有显示声明权限时候也默认使用`default`权限。

3. protected

-   作用范围：可以被同一个包内或者子类访问
-   使用场景：常用语父类只对子类开发，对其他类隐藏的情况。

4. public

-   作用范围:可以被任何类访问
-   使用场景：public修饰符用于那些希望对外公开的类、接口、方法和变量。如果一个类被声明为public，那么该类可以被任何其他类访问。

## 修饰符表格

| 修饰符    | 类内部 | 同一个包 | 子类 | 全局 |
| --------- | ------ | -------- | ---- | ---- |
| private   | 是     | 否       | 否   | 否   |
| default   | 是     | 是       | 否   | 否   |
| protected | 是     | 是       | 是   | 否   |
| public    | 是     | 是       | 是   | 是   |

## 注意

1. 顶层类只有public和default两个修饰符
2. 在选择使用哪种访问修饰符时，一个好的原则是遵循最小权限原则，即只提供必要的访问级别，不必要地提高访问权限。
