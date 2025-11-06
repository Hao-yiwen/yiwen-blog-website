---
title: Kotlin中各个权限关键字
sidebar_label: Kotlin中各个权限关键字
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import kotlin_permisson from '@site/static/img/kotlin_permisson.png'

# Kotlin中各个权限关键字

## open关键字（用于继承）

-   open：在 Kotlin 中，所有的类默认都是 final 的，这意味着它们不能被继承。如果你想允许一个类被继承，你需要使用 open 关键字标记这个类。同样，如果你想允许子类覆盖一个方法或属性，该方法或属性也必须被标记为 open。

## Kotlin 的可见性修饰符

<img src={kotlin_permisson} width={600} />

### public

这是默认的可见性修饰符，如果没有指定可见性修饰符，则默认为 public。public 类或成员可以从任何地方访问。

### private

私有的类或成员只能在定义它们的文件或类中访问。

### protected

Kotlin 和 Java 中都表示成员可以在定义它们的类以及子类中访问。不过在 Kotlin 中，protected 成员在同一个包中的其他类中是不可见的，这一点和 Java 不同，在 Java 中，同一个包中的其他类也可以访问 protected 成员。

### internal

表示该成员只在同一个模块内可见。在 Kotlin 中，模块是一组一起编译的 Kotlin 文件。这通常意味着一个 IntelliJ IDEA 模块、一个 Maven 项目、或者一个 Gradle 源集（source set）。使用 internal 修饰的成员在模块外部是不可见的，这对于隐藏内部实现细节很有用。

## 顶级类可用的可见修饰符

在 Kotlin 中，顶级声明的可见性和 Java 有所不同，提供了更灵活的可见性选项。对于 Kotlin 的顶级类、函数和属性（即直接定义在包下，而非某个类内部的声明），以下是可应用的可见性修饰符及其含义：

-   public（默认）：如果没有显式指定可见性修饰符，顶级声明默认为 public。这意味着任何地方都可以访问该声明，没有任何限制。

-   internal：表示该声明只在同一模块内可见。模块可以是一个 IntelliJ IDEA 模块、一个 Maven 项目、或一个 Gradle 项目等。这比 Java 的默认访问（包级私有）范围要广，但比 public 更加限制。

-   private：表示该声明只在声明它的文件内可见。这允许在同一个文件中定义仅限内部使用的类、函数和属性，而不会影响到其他文件。(定义为 private 的类通常用于实现文件内部逻辑的辅助功能，而不打算在文件外部使用。这有助于保持代码的封装性和模块化，通过减少类的可见性来减少类之间的耦合。)

-   protected：这个修饰符不适用于顶级声明。protected 只能用于类的成员，表示这些成员在定义它们的类及其子类中可见。
