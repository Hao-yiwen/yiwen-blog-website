# 注解

Java 注解（Annotations）是在 Java 5（也称作 Java 1.5，2004年发布）中引入的。注解提供了一种为代码添加元数据的方法，这些元数据可以在编译时、加载时或运行时被读取，从而为程序提供额外的信息。

注解的引入极大地增强了 Java 的表达能力，使得开发者可以通过一种标准化的方式来生成文档、配置代码和进行其他编译时或运行时的处理。注解被广泛应用于各种框架和库中，用于提供配置、数据校验、序列化控制等功能。

## 注解的主要用途包括：

-   编译器指令：注解可以被用作编译器的指令，比如 @Override 注解指示编译器一个方法必须覆盖父类中的方法。
-   编译时和部署时处理：通过注解处理器，可以在编译时生成额外的代码，如 Lombok 库使用注解自动生成 getter 和 setter 方法。
-   运行时处理：注解也可以在运行时被读取，以提供动态的行为配置，这在许多现代 Java 框架中非常常见，如 Spring 和 Hibernate。

## 注解的类型：

-   内置注解：Java 提供了一些预定义的注解，如 @Override、@Deprecated 和 @SuppressWarnings。
-   元注解：用于注解其他注解的注解，如 @Target、@Retention、@Inherited 和 @Documented。
-   自定义注解：开发者可以定义自己的注解来满足特定需求。
-   注解使得 Java 代码更加简洁且易于理解，同时提供了一种强大的工具来支持各种高级编程模式和框架的开发。自从 Java 5 以来，注解已经成为 Java 编程不可或缺的一部分。

## 注解示例代码
[示例代码](https://github.com/Hao-yiwen/java-study/blob/master/javaDemo1/src/main/java/org/example/Greeting.java)

```java
// 定义注解
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

// 指定注解在运行时保留，因此可以通过反射读取
@Retention(RetentionPolicy.RUNTIME)
// 指定注解只能应用于方法上
@Target(ElementType.METHOD)
public @interface Greeting {
    // 定义一个名为 value 的元素，带有默认值
    String value() default "Hello";
}


// 使用注解
public class GreetingExample {

    @Greeting(value = "Hello, World!")
    public void displayGreeting() {
        System.out.println("This is a method with Greeting annotation.");
    }
}

// 读取注解
import java.lang.reflect.Method;

public class AnnotationReader {

    public static void main(String[] args) throws Exception {
        // 获取 GreetingExample 类的 Class 对象
        // 使用反射获取注解内容
        Class<?> cls = Class.forName("GreetingExample");
        // 实例化 GreetingExample
        Object obj = cls.newInstance();
        
        // 遍历 GreetingExample 类的所有方法
        for (Method method : cls.getDeclaredMethods()) {
            // 检查方法上是否有 Greeting 注解
            if (method.isAnnotationPresent(Greeting.class)) {
                // 获取注解实例
                Greeting greeting = method.getAnnotation(Greeting.class);
                // 打印注解的 value 值
                System.out.println("Greeting message: " + greeting.value());
                // 调用方法
                method.invoke(obj);
            }
        }
    }
}
```
