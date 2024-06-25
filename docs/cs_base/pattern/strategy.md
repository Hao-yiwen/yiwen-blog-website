# 策略模式

## 代码

[代码链接](https://github.com/Hao-yiwen/android-study/tree/master/DesignPatterns/src/main/java/org/example/duck)

## 概述

策略模式是一种行为设计模式，它定义了一系列算法，并将每个算法封装起来，使它们可以互换。策略模式使得算法可以在不影响客户端的情况下发生变化。策略模式将行为的定义和实现分离开来，使得系统更加灵活和可扩展。

## 结构

策略模式主要包含以下几个部分：

1.	策略接口 (Strategy)：定义所有支持的算法的公共接口。
2.	具体策略 (Concrete Strategy)：实现策略接口的具体算法。
3.	上下文 (Context)：维护一个对策略对象的引用，并在需要时调用策略对象的方法。

```java
// 策略接口
interface Strategy {
    int doOperation(int num1, int num2);
}

// 具体策略类 - 加法
class OperationAdd implements Strategy {
    @Override
    public int doOperation(int num1, int num2) {
        return num1 + num2;
    }
}

// 具体策略类 - 减法
class OperationSubtract implements Strategy {
    @Override
    public int doOperation(int num1, int num2) {
        return num1 - num2;
    }
}

// 具体策略类 - 乘法
class OperationMultiply implements Strategy {
    @Override
    public int doOperation(int num1, int num2) {
        return num1 * num2;
    }
}

// 上下文类
class Context {
    private Strategy strategy;

    public Context(Strategy strategy) {
        this.strategy = strategy;
    }

    public int executeStrategy(int num1, int num2) {
        return strategy.doOperation(num1, num2);
    }
}

// 客户端代码
public class StrategyPatternDemo {
    public static void main(String[] args) {
        Context context = new Context(new OperationAdd());
        System.out.println("10 + 5 = " + context.executeStrategy(10, 5));

        context = new Context(new OperationSubtract());
        System.out.println("10 - 5 = " + context.executeStrategy(10, 5));

        context = new Context(new OperationMultiply());
        System.out.println("10 * 5 = " + context.executeStrategy(10, 5));
    }
}
```