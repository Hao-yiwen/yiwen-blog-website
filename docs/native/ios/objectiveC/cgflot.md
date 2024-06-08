# 为什么叫CGFloat而不是NSFloat

在 Objective-C 中，CGFloat 是由 Core Graphics 框架定义的，而不是由 Foundation 框架定义的。这就是为什么它叫 CGFloat 而不是 NSFloat。

## Core Graphics 框架和 Foundation 框架

	-	Core Graphics 框架：Core Graphics 是一个 2D 绘图引擎，提供低级别、精确控制的绘图操作，用于图形和图像处理。它通常用于绘制图形、图像和处理图像数据等。Core Graphics 使用 CG 前缀，表示 Core Graphics 定义的类型和函数。
	-	Foundation 框架：Foundation 是一个更通用的框架，提供基础的数据管理、文本处理、日期和时间计算、排序和过滤等功能。Foundation 使用 NS 前缀，表示 NeXTSTEP，这是 macOS 的前身操作系统。

## CGFloat 的定义

CGFloat 是在 Core Graphics 框架中定义的，用于在绘图操作中表示浮点数。根据平台的不同，CGFloat 可以是 float 或 double。

```objectivec
#if defined(__LP64__) && __LP64__
typedef double CGFloat;
#else
typedef float CGFloat;
#endif
```

在 64 位平台上，CGFloat 被定义为 double，而在 32 位平台上，CGFloat 被定义为 float。

### 为什么不是 NSFloat

1.	历史原因：CGFloat 是随着 Core Graphics 框架引入的，而 Core Graphics 是一个独立的低级绘图框架，不属于 Foundation 框架的范畴。
2.	命名约定：Apple 在不同的框架中使用不同的前缀来表示类型的归属。CG 表示 Core Graphics，而 NS 表示 NeXTSTEP（Foundation 框架的前缀）。
3.	平台适应性：CGFloat 的定义使其能够根据平台（32 位或 64 位）自动调整为 float 或 double，提供更好的性能和精度。

### 示例代码
```objectivec
#import <UIKit/UIKit.h>

@interface MyView : UIView
@end

@implementation MyView

- (void)drawRect:(CGRect)rect {
    CGContextRef context = UIGraphicsGetCurrentContext();
    
    CGFloat x = 10.0;
    CGFloat y = 10.0;
    CGFloat width = 100.0;
    CGFloat height = 100.0;
    
    CGRect rectangle = CGRectMake(x, y, width, height);
    
    CGContextSetFillColorWithColor(context, [UIColor blueColor].CGColor);
    CGContextFillRect(context, rectangle);
}

@end
```