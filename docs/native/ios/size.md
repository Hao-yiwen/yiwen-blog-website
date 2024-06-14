# Ios中的尺寸

iOS 的尺寸也是以点（points）为单位，而不是像素（pixels）。这是因为不同的设备可能有不同的像素密度。例如，一个 Retina 屏幕的设备可能有更高的像素密度，这意味着同样的点数会占据更多的像素。

在这个例子中，我们创建了一个宽度为 100，高度为 200 的尺寸。

你可以通过 size.width 和 size.height 来访问尺寸的宽度和高度。例如：

```swift
let size = CGSize(width: 100, height: 200)

print(size.width)  // 输出 "100.0"
print(size.height) // 输出 "200.0"
```

无论是在 Swift、Objective-C 还是在 React Native 中，你都可以使用设备无关的尺寸来创建适应不同屏幕和设备的布局。
