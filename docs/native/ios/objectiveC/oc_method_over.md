---
title: oc的方法重载
sidebar_label: oc的方法重载
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# oc的方法重载

oc不支持java一样的同名方法重载。需要使用不同方法名实现。

```
@interface SampleClass : NSObject
- (void)methodWithNoArgs;
- (void)methodWithOneArg:(int)arg1;
- (void)methodWithTwoArgs:(int)arg1 secondArg:(int)arg2;
@end

@implementation SampleClass
- (void)methodWithNoArgs {
    NSLog(@"No arguments");
}

- (void)methodWithOneArg:(int)arg1 {
    NSLog(@"One argument: %d", arg1);
}

- (void)methodWithTwoArgs:(int)arg1 secondArg:(int)arg2 {
    NSLog(@"Two arguments: %d, %d", arg1, arg2);
}
@end
```

