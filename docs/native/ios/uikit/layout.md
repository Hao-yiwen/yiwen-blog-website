---
title: UIKit常用布局
sidebar_label: UIKit常用布局
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# UIKit常用布局

1. Auto Layout

Auto Layout 是目前 iOS 开发中最常用的布局方式，尤其在处理不同设备尺寸和屏幕旋转时非常有效。它基于约束（constraints）系统，可以在 Interface Builder 中使用，也可以通过代码创建。

- 使用 Interface Builder 设置 Auto Layout

在 Storyboard 或 XIB 文件中，可以使用 Interface Builder 来设置视图的约束（constraints）。

- 代码

```swift
import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let redView = UIView()
        redView.backgroundColor = .red
        redView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(redView)
        
        NSLayoutConstraint.activate([
            redView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            redView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            redView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            redView.heightAnchor.constraint(equalToConstant: 100)
        ])
    }
}
```

2. UIStackView

UIStackView 是在 iOS 9 中引入的一种布局方式，它能够将一组视图按垂直或水平方向堆叠起来，并自动管理这些视图的布局。使用 Stack View 可以简化复杂的视图层次结构。

- 使用 Interface Builder 设置 Stack Views

在 Storyboard 或 XIB 文件中，可以拖动一个 UIStackView 到视图层次结构中，并添加子视图。

- 使用代码设置 Stack Views
```swift
import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let label1 = UILabel()
        label1.text = "Label 1"
        label1.backgroundColor = .yellow
        
        let label2 = UILabel()
        label2.text = "Label 2"
        label2.backgroundColor = .green
        
        let stackView = UIStackView(arrangedSubviews: [label1, label2])
        stackView.axis = .vertical
        stackView.spacing = 10
        stackView.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(stackView)
        
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stackView.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
}
```

3. Frame-based Layout

Frame-based Layout 是最传统的布局方式，通过直接设置视图的 frame 属性来指定其位置和大小。这种方法在需要精确控制视图位置时很有用，但缺乏自动调整能力。

```swift
import UIKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let redView = UIView()
        redView.backgroundColor = .red
        redView.frame = CGRect(x: 20, y: 50, width: 200, height: 100)
        view.addSubview(redView)
    }
}
```