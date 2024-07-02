# LinkedList

双向链表列表。

在 Java 中，LinkedList 是 List 接口的一个具体实现，继承自 AbstractSequentialList，并实现了 List、Deque 和 Queue 接口。

```java
import java.util.LinkedList;

public class Main {
    public static void main(String[] args) {
        LinkedList<String> list = new LinkedList<>();

        // 添加元素
        list.add("Element 1");
        list.add("Element 2");
        list.addFirst("Element 0");
        list.addLast("Element 3");

        // 遍历元素
        for (String element : list) {
            System.out.println(element);
        }

        // 删除元素
        list.removeFirst();
        list.removeLast();
        list.remove("Element 2");

        // 获取元素
        System.out.println("First element: " + list.getFirst());
        System.out.println("Last element: " + list.getLast());
    }
}
```

## 应用场景

-   插入和删除频繁的场景：链表在插入和删除操作频繁的场景中表现出色。
-   FIFO（先进先出）队列：使用链表可以轻松实现队列数据结构。
-   需要动态扩展的场景：链表可以根据需要动态扩展，不受预先分配大小的限制。
