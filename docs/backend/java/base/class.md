---
sidebar_position: 10
---

# java中的类

## 普通类（Regular Class）

1. 特点：可以包含属性、方法、构造器等。
2. 解决的问题：定义具体对象的属性和行为，如汽车的颜色和行驶功能。
```java
public class Car {
    private String color;

    public Car(String color) {
        this.color = color;
    }

    public void drive() {
        System.out.println("Driving a " + color + " car.");
    }
}
```

## 抽象类（Abstract Class）
1. 特点：可以包含属性、方法、构造器等。
2. 解决的问题：定义具体对象的属性和行为，如汽车的颜色和行驶功能。
```java
public abstract class Animal {
    private String name;

    public Animal(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    // 抽象方法，由子类具体实现
    public abstract void makeSound();

    public void eat() {
        System.out.println(name + " is eating.");
    }
}

public class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }

    @Override
    public void makeSound() {
        System.out.println(getName() + " says: Woof!");
    }
}

public class Cat extends Animal {
    public Cat(String name) {
        super(name);
    }

    @Override
    public void makeSound() {
        System.out.println(getName() + " says: Meow!");
    }
}
```

## 最终类（Final Class）
1. 特点：不能被继承，保证类的不变性。
2. 解决的问题：创建不变的类，如常量集合或实用程序类。
```java
public final class Constants {
    public static final double PI = 3.14159;
}
```

## 接口（Interface）
1. 特点：定义方法签名，Java 8后可以包含默认方法和静态方法。
2. 解决的问题：定义一组行为规范，使不同的类可以实现共同的接口。
```java
interface MyInterface {
    // java8新增默认方法
    default void showDefault() {
        System.out.println("This is a default method");
    }

    // 静态方法
    static void showStatic() {
        System.out.println("This is a static method");
    }

    // 常量
    String INTERFACE_CONSTANT = "CONSTANT_VALUE";

    // 普通方法声明
    public void finish();
}
```

## 枚举类（Enum Class）
1. 特点：定义一组命名的常量。
2. 解决的问题：定义一组固定的值（如星期天），更安全和易于维护。
```java
public enum Day {
    MONDAY(1), TUESDAY(2), WEDNESDAY(3), THURSDAY(4), FRIDAY(5), SATURDAY(6), SUNDAY(7);

    private final int value;

    private Day(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
```

:::tip
在Java中，使用 enum 关键字声明的枚举本身就代表一种独特的类型。当您声明一个枚举如 Day，它实际上是定义了一个名为 Day 的新的类型，这个类型具有一组固定的实例（在您的例子中是 MONDAY, TUESDAY, WEDNESDAY 等）。

每个枚举常量（如 Day.MONDAY）都是 Day 类型的一个实例。这使得枚举在Java中非常强大和灵活，因为它们不仅仅是简单的值（像在许多其他编程语言中那样），而是可以携带自己的行为和状态的完整对象。
:::

## 匿名类（Anonymous Class）
1. 特点：用于创建一次性使用的类。
2. 解决的问题：快速实现接口或扩展类，通常用于事件监听和回调。
```java
new Thread(new Runnable() {
    @Override
    public void run() {
        System.out.println("Hello from anonymous class.");
    }
}).start();
```
:::tip
所以，“匿名类”这个术语指的是类定义本身没有名称，而非其实例。您完全可以像操作其他对象一样，给这个匿名类的实例命名，并在代码中使用这个命名的引用。这种方式在需要实现接口或继承类的同时，又不想创建一个完整的新类时非常有用。
:::

## 内部类（Inner Class）
1. 特点：定义在另一个类的内部，可以访问外部类的成员。
2. 解决的问题：逻辑上属于外部类的一部分，用于更好的封装。
```java
// 假设我们有一个表示购物车的类，购物车中包含多个商品项。每个商品项可以看作是购物车的一部分，与购物车紧密相关联。在这种情况下，将商品项作为内部类定义在购物车类中可以提高封装性，并且允许商品项直接访问购物车的私有成员。
public class ShoppingCart {
    private List<Item> items = new ArrayList<>();

    // 内部类：商品项
    public class Item {
        private String name;
        private double price;

        public Item(String name, double price) {
            this.name = name;
            this.price = price;
        }

        public void display() {
            System.out.println(name + ": $" + price);
        }

        public double getPrice() {
            return price;
        }
    }

    public void addItem(String name, double price) {
        Item newItem = new Item(name, price);
        items.add(newItem);
    }

    public double calculateTotal() {
        double total = 0;
        for (Item item : items) {
            total += item.getPrice();
        }
        return total;
    }

    public void displayItems() {
        for (Item item : items) {
            item.display();
        }
    }
}
```

## 静态嵌套类（Static Nested Class）
1. 特点：像静态成员，不需要外部类的实例。
2. 解决的问题：静态嵌套类（Static Nested Class）在Java中是定义在另一个类内部的静态类。它们通常用于当类与另一个类紧密相关，但不需要访问外部类的实例成员时。使用静态嵌套类可以使代码更加模块化，并且有助于将相关的类放在一起，增强封装性。
```java
// 示例：网络请求和响应处理
// 假设我们有一个处理网络请求的类，这个类可能包含多个与网络请求相关的静态嵌套类，如请求构建器、响应处理器等。这些嵌套的静态类与网络请求处理紧密相关，但不需要访问网络请求处理类的实例成员。
public class NetworkClient {
    // 静态嵌套类：请求构建器
    public static class RequestBuilder {
        private String url;
        private String method;

        public RequestBuilder setUrl(String url) {
            this.url = url;
            return this;
        }

        public RequestBuilder setMethod(String method) {
            this.method = method;
            return this;
        }

        public void send() {
            System.out.println("Sending " + method + " request to " + url);
            // 实际的发送逻辑
        }
    }

    // 静态嵌套类：响应处理器
    public static class ResponseHandler {
        public void handleResponse(String response) {
            System.out.println("Handling response: " + response);
            // 实际的响应处理逻辑
        }
    }
}

public class Main {
    public static void main(String[] args) {
        // 使用请求构建器发送请求
        NetworkClient.RequestBuilder builder = new NetworkClient.RequestBuilder();
        builder.setUrl("http://example.com").setMethod("GET").send();

        // 使用响应处理器处理响应
        NetworkClient.ResponseHandler handler = new NetworkClient.ResponseHandler();
        handler.handleResponse("Response from http://example.com");
    }
}
```

:::info
静态嵌套类（Static Nested Class）和内部类（Inner Class）确实可以被普通类（Regular Class）代替实现，但使用嵌套类（无论是静态的还是非静态的）主要是出于封装性和模块化的考虑。
:::