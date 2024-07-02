# 适配器模式

适配器模式（Adapter Pattern）是一种设计模式，它用于将一个类的接口转换成客户期望的另一个接口。适配器模式使得原本由于接口不兼容而不能一起工作的类可以一起工作。该模式主要应用于希望复用一些现存的类，但是接口又与复用环境要求的接口不一致的情况。

## 适配器模式的示例

下面是一个简单的 Java 示例，展示如何使用适配器模式将一个类的接口转换成另一个接口。

假设你有一个现有的类 OldSystem，它有一个方法 specificRequest，但你需要使用一个新接口 Target，它定义了一个方法 request。

### 定义目标接口
```java
public interface Target {
    void request();
}
```

### 现有类
```java
public class OldSystem {
    public void specificRequest() {
        System.out.println("Called specificRequest()");
    }
}
```

### 适配器类
```java
public class Adapter implements Target {
    private OldSystem oldSystem;

    public Adapter(OldSystem oldSystem) {
        this.oldSystem = oldSystem;
    }

    @Override
    public void request() {
        oldSystem.specificRequest();
    }
}
```

### 客户端代码
```java
public class Client {
    public static void main(String[] args) {
        OldSystem oldSystem = new OldSystem();
        Target target = new Adapter(oldSystem);
        target.request();
    }
}
```

## 解释

1.	目标接口 (Target)：定义了客户端需要的接口。
2.	现有类 (OldSystem)：这是一个现有的类，接口不符合目标接口。
3.	适配器 (Adapter)：实现目标接口，并在其方法中调用现有类的方法。
4.	客户端 (Client)：通过目标接口使用适配器，而不直接使用现有类。

## 使用场景

适配器模式适用于以下场景：

-	当你希望使用一个已经存在的类，但它的接口不符合你的需求时。
-	当你想创建一个可以复用的类，并让它与将来不兼容的类兼容时。
-	当你希望封装类的多个版本（例如：封装遗留代码）时。