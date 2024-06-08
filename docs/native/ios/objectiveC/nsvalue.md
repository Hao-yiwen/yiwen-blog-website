# NSValue

NSValue 是 Objective-C 中的一个类，用于包装和存储基础数据类型和结构体，使它们能够在集合类（如 NSArray、NSDictionary 等）中使用。NSValue 可以存储任意类型的值，包括基本数据类型（如 int、float 等）、结构体（如 CGRect、CGSize、CGPoint 等）以及指针。

## NSValue 常用场景

1.	在集合类中存储非对象类型
2.	传递结构体数据
3.	桥接 C 语言和 Objective-C 代码
4.	使用动画和手势

## 具体示例

1. 在集合类中存储非对象类型

你无法直接在 NSArray 中存储基本数据类型或结构体类型，因此可以使用 NSValue 来包装这些类型。

示例：存储 CGRect 到 NSArray 中
```objectivec
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // 创建一个包含 CGRect 的 NSArray
        NSArray<NSValue *> *rectArray = @[
            [NSValue valueWithCGRect:CGRectMake(0, 0, 100, 100)],
            [NSValue valueWithCGRect:CGRectMake(10, 10, 200, 200)],
            [NSValue valueWithCGRect:CGRectMake(20, 20, 300, 300)]
        ];
        
        // 遍历数组并提取 CGRect
        for (NSValue *value in rectArray) {
            CGRect rect = [value CGRectValue];
            NSLog(@"Rect: %@", NSStringFromCGRect(rect));
        }
    }
    return 0;
}
```

2. 传递结构体数据

在方法或函数之间传递复杂的结构体数据时，可以使用 NSValue 包装这些结构体。

```objectivec
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface MyClass : NSObject

- (void)printPoint:(NSValue *)pointValue;

@end

@implementation MyClass

- (void)printPoint:(NSValue *)pointValue {
    CGPoint point = [pointValue CGPointValue];
    NSLog(@"Point: %@", NSStringFromCGPoint(point));
}

@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        MyClass *myClass = [[MyClass alloc] init];
        CGPoint point = CGPointMake(10, 20);
        [myClass printPoint:[NSValue valueWithCGPoint:point]];
    }
    return 0;
}
```

3. 桥接 C 语言和 Objective-C 代码

在混合使用 C 和 Objective-C 代码时，可以使用 NSValue 来包装和传递 C 语言的结构体或指针。

示例：包装 void * 指针
```objectivec
#import <Foundation/Foundation.h>

void myFunction(void *ptr) {
    NSLog(@"Pointer: %p", ptr);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        int value = 42;
        NSValue *pointerValue = [NSValue valueWithPointer:&value];
        myFunction([pointerValue pointerValue]);
    }
    return 0;
}
```