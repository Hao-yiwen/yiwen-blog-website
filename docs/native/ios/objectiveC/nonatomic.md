# 定义成员变量的时候需要指定nonatomic和strong，两者分别是干什么的

```oc
@interface SampleClass : NSObject
@property (nonatomic, strong) NSString *name;
@end

@implementation SampleClass
@end
```

-   第一个属性介绍: nonatomic 和 atomic 的区别

1. 默认情况下，属性是 atomic 的，这意味着编译器会生成线程安全的访问器方法，以确保属性在多线程环境下的读写操作是安全的。然而，这种线程安全性是通过加锁实现的，会带来一定的性能开销。

2. 使用 nonatomic 可以避免这种性能开销，因为它不会生成线程安全的访问器方法。

-   第二个属性介绍: strong 和 weak 的区别

在 Objective-C 中，strong 是一个属性修饰符，用于指定属性对其所引用的对象持有强引用（strong reference）。强引用确保引用的对象在属性持有它期间不会被释放。这是自动引用计数（ARC）的一部分，ARC 是用于内存管理的一种机制。

1.	strong：表示强引用。持有强引用的对象在引用期间不会被销毁。当没有任何强引用指向该对象时，该对象才会被释放。
2.	weak：表示弱引用。持有弱引用的对象不会影响对象的引用计数，引用的对象可能会在任意时刻被释放。弱引用用于避免循环引用（retain cycles），特别是在对象之间存在相互引用的情况下。
