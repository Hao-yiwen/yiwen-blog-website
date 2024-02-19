# instanceof

## 介绍

instanceof 是 Java 中的一个关键字，用于测试一个对象实例是否是特定类型或该类型的子类型。具体来说，它用于检查左侧的对象是否是右侧类/接口的实例，或者是否是其子类/实现类的实例。如果是，instanceof 操作返回 true；否则返回 false。

## 重要

instanceof 关键字不仅可以用于判断一个对象是否是某个具体类的实例，还可以用来判断该对象是否是一个特定接口的实现。这使得 instanceof 在检查对象是否符合某个接口类型时非常有用，无论该对象属于哪个类，只要它实现了指定的接口，instanceof 操作就会返回 true。

## 使用场景

instanceof 在需要判断对象具体类型或确认对象是否能够按照特定类型处理的情况下非常有用。它常用于以下场景：

-   在向下转型（casting）之前检查对象类型，避免 ClassCastException。
-   在实现方法的多态性时，判断传入对象的具体类型。
-   在处理集成自同一基类的不同子类对象时，确定具体实例类型。

## 示例

```java
// 判断是否是某个类的子类
class Animal {}
class Dog extends Animal {}
class Cat extends Animal {}


Animal myDog = new Dog();
Animal myCat = new Cat();

System.out.println(myDog instanceof Dog); // 输出 true
System.out.println(myDog instanceof Cat); // 输出 false
System.out.println(myDog instanceof Animal); // 输出 true
System.out.println(myCat instanceof Animal); // 输出 true

// 判断是否是某个接口的实现
interface Shape {
    void draw();
}

class Circle implements Shape {
    public void draw() {
        System.out.println("Drawing Circle");
    }
}

class Rectangle implements Shape {
    public void draw() {
        System.out.println("Drawing Rectangle");
    }
}

Shape circle = new Circle();
Shape rectangle = new Rectangle();

System.out.println(circle instanceof Shape); // 输出 true
System.out.println(rectangle instanceof Shape); // 输出 true
```
