---
title: OC一个接口是不是只能有一个类名
sidebar_label: OC一个接口是不是只能有一个类名
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# OC一个接口是不是只能有一个类名

1. oc的@interface不是创建接口，而是类声明，实现是@implemetion
2. oc中的接口用@protocol实现

```oc
#import <Foundation/Foundation.h>

@protocol Animal <NSObject>
- (void)makeSound;
@end

@interface Dog : NSObject <Animal>
@end

@implementation Dog
- (void)makeSound {
    NSLog(@"Woof");
}
@end

@interface Cat : NSObject <Animal>
@end

@implementation Cat
- (void)makeSound {
    NSLog(@"Meow");
}
@end
```