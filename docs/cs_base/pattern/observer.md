---
title: 观察者模式
sidebar_label: 观察者模式
date: 2024-06-25
last_update:
  date: 2024-06-26
---

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

## 示例代码

```java
import java.util.ArrayList;
import java.util.List;

// 主题接口
interface Subject {
    void registerObserver(Observer o);
    void removeObserver(Observer o);
    void notifyObservers();
}

// 具体主题类
class ConcreteSubject implements Subject {
    private List<Observer> observers;
    private int state;

    public ConcreteSubject() {
        this.observers = new ArrayList<>();
    }

    public int getState() {
        return state;
    }

    public void setState(int state) {
        this.state = state;
        notifyObservers();
    }

    @Override
    public void registerObserver(Observer o) {
        observers.add(o);
    }

    @Override
    public void removeObserver(Observer o) {
        observers.remove(o);
    }

    @Override
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update();
        }
    }
}

// 观察者接口
interface Observer {
    void update();
}

// 具体观察者类
class ConcreteObserver implements Observer {
    private ConcreteSubject subject;

    public ConcreteObserver(ConcreteSubject subject) {
        this.subject = subject;
        this.subject.registerObserver(this);
    }

    @Override
    public void update() {
        System.out.println("Observer notified. State is: " + subject.getState());
    }
}

// 客户端代码
public class ObserverPatternDemo {
    public static void main(String[] args) {
        ConcreteSubject subject = new ConcreteSubject();

        new ConcreteObserver(subject);
        new ConcreteObserver(subject);

        System.out.println("First state change: 15");
        subject.setState(15);

        System.out.println("Second state change: 10");
        subject.setState(10);
    }
}
```