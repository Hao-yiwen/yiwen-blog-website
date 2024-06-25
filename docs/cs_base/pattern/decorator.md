# 装饰器模式


装饰器模式（Decorator Pattern）是一种结构型设计模式，用于动态地向对象添加职责（功能）。装饰器模式允许通过将对象放入包含行为的特殊封装对象中来为对象添加功能。与继承不同，装饰器模式在运行时增加对象的功能，而不是在编译时通过扩展类来实现。

以下是装饰器模式的主要特点和实现方法：

## 代码

[代码](https://github.com/Hao-yiwen/android-study/tree/master/DesignPatterns/src/main/java/org/example/decorator)

## 主要特点

1.	灵活性：可以在运行时选择不同的装饰器组合来改变对象的行为。
2.	透明性：客户端代码可以透明地使用装饰过的对象，而无需改变其代码。
3.	可组合性：多个装饰器可以结合在一起，以创造出复杂的行为。

## 组成部分

1.	组件接口（Component）：定义对象的接口，可以被装饰器扩展。
2.	具体组件（ConcreteComponent）：实现组件接口的类，是被装饰器装饰的原始对象。
3.	装饰器（Decorator）：实现组件接口，并且持有一个指向组件对象的引用，可以在该对象的基础上增加行为。
4.	具体装饰器（ConcreteDecorator）：继承自装饰器类，负责增加具体的行为。

```java
// Component 接口
interface Component {
    String operation();
}

// ConcreteComponent 类
class ConcreteComponent implements Component {
    @Override
    public String operation() {
        return "ConcreteComponent";
    }
}

// Decorator 抽象类
abstract class Decorator implements Component {
    protected Component component;

    public Decorator(Component component) {
        this.component = component;
    }

    @Override
    public String operation() {
        return component.operation();
    }
}

// ConcreteDecoratorA 类
class ConcreteDecoratorA extends Decorator {
    public ConcreteDecoratorA(Component component) {
        super(component);
    }

    @Override
    public String operation() {
        return "ConcreteDecoratorA(" + super.operation() + ")";
    }
}

// ConcreteDecoratorB 类
class ConcreteDecoratorB extends Decorator {
    public ConcreteDecoratorB(Component component) {
        super(component);
    }

    @Override
    public String operation() {
        return "ConcreteDecoratorB(" + super.operation() + ")";
    }
}

// 客户端代码
public class DecoratorPatternDemo {
    public static void main(String[] args) {
        Component simple = new ConcreteComponent();
        System.out.println("Client: I've got a simple component:");
        System.out.println("RESULT: " + simple.operation());

        Component decorator1 = new ConcreteDecoratorA(simple);
        Component decorator2 = new ConcreteDecoratorB(decorator1);
        System.out.println("Client: Now I've got a decorated component:");
        System.out.println("RESULT: " + decorator2.operation());
    }
}
```