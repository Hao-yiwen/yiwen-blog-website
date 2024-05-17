# 依赖注入(DI)

[参考文档](https://juejin.cn/post/6857406008877121550)

## 介绍

依赖注入（Dependency Injection, DI）是一种软件设计模式，旨在实现控制反转（Inversion of Control, IoC）的原则，以提高代码的模块化和可测试性。在这种模式下，对象的依赖不是由对象本身创建，而是由外部容器或框架在创建对象时注入。这样做的目的是减少组件之间的耦合，使得代码更加灵活、易于管理和扩展。

## Spring中的依赖注入

Spring框架是Java平台上广泛使用的开源应用程序框架之一，它提供了全面的编程和配置模型。Spring的核心特性之一是依赖注入（DI），这是一种实现控制反转（IoC）的方法。在Spring中，对象之间的依赖关系通过框架在运行时自动建立，而不是由对象本身在编译时静态定义。

### 示例代码

[示例代码](https://github.com/Hao-yiwen/java-study)

## spring中的依赖注入方式

-   构造函数注入：通过类的构造函数注入依赖，适用于必需的依赖，确保所需依赖的不变性。

```java
public class ExampleBean {
    private AnotherBean beanOne;
    private YetAnotherBean beanTwo;

    public ExampleBean(AnotherBean beanOne, YetAnotherBean beanTwo) {
        this.beanOne = beanOne;
        this.beanTwo = beanTwo;
    }
}
```

-   Setter注入：通过类的setter方法注入依赖，适用于可选依赖或需要重新配置依赖的情况。

```java
public class ExampleBean {
    private AnotherBean beanOne;

    public void setBeanOne(AnotherBean beanOne) {
        this.beanOne = beanOne;
    }
}
```

-   字段注入：直接在类的字段上注入依赖，虽然简便，但不推荐使用，因为它增加了类和Spring框架之间的耦合，并且可能会影响测试性和清晰度。

```java
public class ExampleBean {
    @Autowired
    private AnotherBean beanOne;
}
```

## spring中的依赖配置方式

-   基于注解的配置：使用@Autowired、@Inject、@Resource等注解标记要注入的依赖。

-   基于XML的配置：在XML文件中声明bean及其依赖关系。

```xml
<beans>
    <bean id="beanOne" class="com.example.AnotherBean"/>
    <bean id="exampleBean" class="com.example.ExampleBean">
        <property name="beanOne" ref="beanOne"/>
    </bean>
</beans>
```
