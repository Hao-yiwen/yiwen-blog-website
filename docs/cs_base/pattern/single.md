# 单例模式

单例模式（Singleton Pattern）是一种创建型设计模式，旨在确保一个类只有一个实例，并提供一个全局访问点来访问该实例。单例模式常用于需要全局唯一对象的场景，如配置类、日志记录类、线程池等。

## 单例模式的主要特点

1.	唯一性：单例模式确保一个类只有一个实例。
2.	全局访问点：提供一个全局访问点，使得外部可以访问该实例。

## 实现单例模式的步骤

1.	私有化构造函数：防止外部通过构造函数创建实例。
2.	创建一个私有的静态实例：在类内部创建一个私有的静态实例变量。
3.	提供一个公有的静态方法：提供一个公有的静态方法，用于返回类的唯一实例。

```java
package org.example.single;

public class ChocolateBoiler {
    private volatile static ChocolateBoiler instance;
    private boolean empty;
    private boolean boild;

    private ChocolateBoiler(){
        this.empty = true;
        this.boild = false;
    }

    public static ChocolateBoiler getInstance(){
        if(instance == null){
            synchronized (ChocolateBoiler.class){
                if(instance == null){
                    instance = new ChocolateBoiler();
                }
            }
        }
        return instance;
    }

    public void drain(){
        if(!isEmpty() && isBoild()){
            this.empty = true;
        }
    }

    public void fill(){
        if(isEmpty()){
            this.empty = false;
            this.boild = false;
        }
    }

    public boolean isEmpty(){
        return this.empty;
    }

    public void boil(){
        if(!isEmpty() && !isBoild()){
            this.boild = true;
        }
    }

    public boolean isBoild(){
        return this.boild;
    }
}
```