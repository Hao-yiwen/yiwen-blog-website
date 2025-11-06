---
title: Object在Kt中的使用
sidebar_label: Object在Kt中的使用
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Object在Kt中的使用

object在kt中使用广泛，例如有

-   使用匿名内部类实现函数接口声明

```kt
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

-   单例模式

```kt
// 使用object实现
object singleTon{
    var number = 0
    fun increase(){
        number++
    }
}

fun mian(){
    val number1 = singleTon.number
    println(number1) // 0

    singleTon.increase()
    println(singleTon.number)  // 1
    singleTon.increase()

    val number2 = singleTon.number
    println(number2) //2
}

// 使用私有类实现
class SingleTon private constructor(){
    var number:Int = 0

    fun increase(){
        number++
    }

    companion object{
        /**
            * @description: 确保对所有线程的可见性
            */
        @Volatile
        private var instance: Singleton1? = null

        /**
            * @description: 双重校验锁式
            */
        fun getInstance(): Singleton1 = instance ?: synchronized(this) {
            instance ?: Singleton1().also { instance = it }
        }
    }
}

fun main(){
    val singleton1 = Singleton1.getInstance()

    println(singleton1.number) // 0

    singleton1.increase()

    println(singleton1.number) // 1

    singleton1.increase()

    val singleton2 = Singleton1.getInstance()

    println(singleton2.number) // 2
}
```

-   伴生对象

相当月java的中静态成员和方法。

```kt
class MyClass{
    companion object {
        final val number = 0;
    }
}
```

```java
class MyClass{
    static final int number = 0;
}
```
