---
title: 工厂模式
sidebar_label: 工厂模式
date: 2024-06-26
last_update:
  date: 2024-06-26
---

# 工厂模式

工厂模式（Factory Pattern）是一种创建型设计模式，主要用于在实例化对象时隐藏复杂性。通过定义一个用于创建对象的接口，而不是直接实例化对象，工厂模式可以让代码更具可扩展性和可维护性。

## 代码

[代码](https://github.com/Hao-yiwen/android-study/tree/master/DesignPatterns/src/main/java/org/example/factory)

## 主要特点

1.	解耦：将对象的创建与其使用解耦，用户无需知道具体的创建过程。
2.	单一职责：创建逻辑集中在一个地方，更易于维护和修改。
3.	可扩展性：通过引入新的工厂类，轻松添加新的产品类型。

## 工厂模式的类型

工厂模式有几种变体，每种都有其特定的用例和实现方式：

1.	简单工厂（Simple Factory）：通过一个工厂类的静态方法来创建对象，通常用于创建单一类型的对象。
2.	工厂方法（Factory Method）：定义一个接口或抽象类用于创建对象，具体的子类决定实例化哪个类。每个子类都有其专门的工厂方法。
3.	抽象工厂（Abstract Factory）：提供一个接口，用于创建相关或依赖对象的家族，而无需明确指定具体类。

## 示例代码

```java title="简单工厂方法"
// 产品接口
interface Product {
    void use();
}

// 具体产品A
class ConcreteProductA implements Product {
    public void use() {
        System.out.println("Using Product A");
    }
}

// 具体产品B
class ConcreteProductB implements Product {
    public void use() {
        System.out.println("Using Product B");
    }
}

// 简单工厂类
class SimpleFactory {
    public static Product createProduct(String type) {
        if (type.equals("A")) {
            return new ConcreteProductA();
        } else if (type.equals("B")) {
            return new ConcreteProductB();
        }
        throw new IllegalArgumentException("Unknown product type");
    }
}

// 客户端代码
public class Client {
    public static void main(String[] args) {
        Product productA = SimpleFactory.createProduct("A");
        productA.use();
        
        Product productB = SimpleFactory.createProduct("B");
        productB.use();
    }
}
```

```java title="工厂模式"
// 产品接口
interface Product {
    void use();
}

// 具体产品A
class ConcreteProductA implements Product {
    public void use() {
        System.out.println("Using Product A");
    }
}

// 具体产品B
class ConcreteProductB implements Product {
    public void use() {
        System.out.println("Using Product B");
    }
}

// 抽象工厂
abstract class Factory {
    public abstract Product createProduct();
}

// 具体工厂A
class ConcreteFactoryA extends Factory {
    public Product createProduct() {
        return new ConcreteProductA();
    }
}

// 具体工厂B
class ConcreteFactoryB extends Factory {
    public Product createProduct() {
        return new ConcreteProductB();
    }
}

// 客户端代码
public class Client {
    public static void main(String[] args) {
        Factory factoryA = new ConcreteFactoryA();
        Product productA = factoryA.createProduct();
        productA.use();
        
        Factory factoryB = new ConcreteFactoryB();
        Product productB = factoryB.createProduct();
        productB.use();
    }
}
```

```java title="抽象工厂"
// 抽象产品A
interface AbstractProductA {
    void use();
}

// 抽象产品B
interface AbstractProductB {
    void eat();
}

// 具体产品A1
class ConcreteProductA1 implements AbstractProductA {
    public void use() {
        System.out.println("Using Product A1");
    }
}

// 具体产品B1
class ConcreteProductB1 implements AbstractProductB {
    public void eat() {
        System.out.println("Eating Product B1");
    }
}

// 抽象工厂
interface AbstractFactory {
    AbstractProductA createProductA();
    AbstractProductB createProductB();
}

// 具体工厂1
class ConcreteFactory1 implements AbstractFactory {
    public AbstractProductA createProductA() {
        return new ConcreteProductA1();
    }

    public AbstractProductB createProductB() {
        return new ConcreteProductB1();
    }
}

// 客户端代码
public class Client {
    public static void main(String[] args) {
        AbstractFactory factory = new ConcreteFactory1();
        AbstractProductA productA = factory.createProductA();
        AbstractProductB productB = factory.createProductB();
        productA.use();
        productB.eat();
    }
}
```