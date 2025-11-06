---
title: java和kt中lambda实现及原始实现
sidebar_label: java和kt中lambda实现及原始实现
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# java和kt中lambda实现及原始实现

## Kotlin

```kt
// 使用lambda表达式
fun mutilyFun(){
    val mutily = {x:Int,y:Int ->
        x*y
    }
    print(mutily(1,2)) // 2
}

// 不使用lambda表达式，使用函数式接口和匿名内部类
fun interface mutilyInteface{
    fun mutily(x:Int,y:Int):Int
}

fun mutilyFun(){
    val mutily = object: mutilyInteface{
        override fun mutily(x:Int,y:Int):Int{
            return x*y
        }
    }
    print(mutily(1,2)) // 2
}
```

## Java

```java
// 使用匿名内部类
 @Test
public void mutilyFun() {
    IntBinaryOperator intUnaryOperator = (x, y) -> {
        return x * y;
    };

    System.out.println(intUnaryOperator.applyAsInt(10, 20));
}


// 使用匿名内部类
interface MutilyInterface{
    int mutily(int x, int y);
}

@Test
public void mutilyFun(){
    MutilyInterface mutilyInterface = new MutilyInterface() {
        @Override
        public int mutily(int x, int y) {
            return x * y;
        }
    };

    System.out.println(mutilyInterface.mutily(10, 20));
}
```
