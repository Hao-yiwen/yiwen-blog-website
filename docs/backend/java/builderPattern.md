---
sidebar_position: 11
---

# 建造者模式

1. 分离构建和表示：建造者模式将对象的构建（即创建对象的过程）和对象的表示（即对象的最终结构）分离开来。这使得同一个构建过程可以创建不同的表示。

2. 链式调用：建造者模式通常提供链式调用的API。这意味着你可以连续调用设置方法（setter methods），每个方法返回当前对象的引用（通常是 this），从而实现流畅的接口。

3. 自我实例化：在建造者模式中，通常由一个专门的建造者（Builder）类来负责构建最终对象。这个建造者类通常不是自我实例化的；相反，它是由客户端代码（即使用这个模式的代码）实例化的。

4. 逐步构建：建造者模式允许逐步构建复杂对象，添加必要的组件。最终，使用一个特定的方法（如 build() 或 create()）来完成对象的构建并获取最终对象。

```java title="示例"
// 考虑一个简单的 Pizza 类和一个对应的 PizzaBuilder 类：
public class Pizza {
    private String dough;
    private String sauce;
    private String topping;

    public Pizza(String dough, String sauce, String topping) {
        this.dough = dough;
        this.sauce = sauce;
        this.topping = topping;
    }

    // Getters and setters

    public static class PizzaBuilder {
        private String dough = "";
        private String sauce = "";
        private String topping = "";

        public PizzaBuilder setDough(String dough) {
            this.dough = dough;
            return this;
        }

        public PizzaBuilder setSauce(String sauce) {
            this.sauce = sauce;
            return this;
        }

        public PizzaBuilder setTopping(String topping) {
            this.topping = topping;
            return this;
        }

        public Pizza build() {
            return new Pizza(dough, sauce, topping);
        }
    }
}

// 使用
Pizza pizza = new Pizza.PizzaBuilder()
                .setDough("Thin Crust")
                .setSauce("Tomato")
                .setTopping("Cheese")
                .build();
```