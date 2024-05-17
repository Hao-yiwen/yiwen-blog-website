# ? super和? extends区别

## ? extends T（上界通配符）

-   **意义：**表示可以是T类型或T的子类型。这种形式被称为"协变"。
-   **用途：**主要用于安全地读取T类型的数据。因为你知道集合中的任何元素至少是T类型的实例，所以可以安全地将它们读取为T类型。
-   **限制：**你不能向使用? extends T的集合中添加任何元素（除了null），因为你不能保证列表实际上是哪个具体子类型的列表。例如，`List<? extends Animal>`可以引用List`<Dog>`，也可以引用`List<Cat>`，你不能向其中添加一个Dog或Cat对象，因为这会破坏类型安全。

## ? super T（下界通配符）

-   **意义：**表示可以是T类型或T的父类型。这种形式被称为"逆变"。
-   **用途：**主要用于安全地写入T类型的数据。因为集合被声明可以包含T类型，所以向其中添加T或其子类型的实例是安全的。
-   **限制：**读取时，你不能保证得到的是T类型的对象。由于集合可以包含T的任何父类型，所以最安全的读取类型只能是Object。

## 两者的限制区别

-   ? extends T允许你读取T类型的数据，但不允许你（安全地）添加数据（除了null）。
-   ? super T允许你添加T类型的数据，但读取数据时只能保证它们是Object类型。

## 看一段代码

```java
public class GenericsLearning {
    class Animal {
        public void eat() {
            System.out.println("Animal is eating");
        }
    }

    class Dog extends Animal {

        public void eat() {
            System.out.println("Dog is eating");
        }

        public void bark() {
            System.out.println("Dog is barking");
        }
    }

    class Cat extends Animal {
        public void eat() {
            System.out.println("Cat is eating");
        }

        public void meow() {
            System.out.println("Cat is meowing");
        }
    }

    public void main(String[] args) {
        List<? super Dog> list = new ArrayList<Animal>();
        list.add(new Animal()); // 为什么这里会报错
    }
}
```

正确解释：因为`? super T`可以添加T及其子类，例如`Animal`中会添加`Cat`和`Dog`类，此时?属于`? super T`类型,所以需要用`? super T`来表示对不对。虽然用`new ArrayList<Animal>()`来表示该类型，但是实际上该类型是`Object`。
