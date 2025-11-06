---
title: 使用navigation导航时没有导航头但是能左滑
sidebar_label: 使用navigation导航时没有导航头但是能左滑
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 使用navigation导航时没有导航头但是能左滑

## 场景说明

- 在开发ios应用的时候有一个场景是创建一个storyboard，然后从首页跳转过去，我的做法是swiftui外层包一个viewcontroller，然后使用storyboard refrence放在子模块中。

- 因为swftui有自己的导航模式，所以我希望在导航到swiftui的时候，导航栏隐藏，然后左滑返回又能出现。（本来以为只是简单的设置navigationbar隐藏和显示）

## 文档

[解决方案](https://stackoverflow.com/questions/24710258/no-swipe-back-when-hiding-navigation-bar-in-uinavigationcontroller)


## 示例代码

```swift
// 创建新的交互手势识别委托实例
import UIKit

class InteractivePopRecognizer: NSObject, UIGestureRecognizerDelegate {

    weak var navigationController: UINavigationController?

    init(controller: UINavigationController) {
        self.navigationController = controller
    }

    func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
        return (navigationController?.viewControllers.count)! > 1
    }

    // 左滑手势识别
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
}

import UIKit
import SwiftUI

public class SwiftUIController: UIViewController, UIGestureRecognizerDelegate {
    var popRecognizer: InteractivePopRecognizer?
    
    public override func viewDidLoad() {
        super.viewDidLoad()
        
        let swiftSimpleView = SwiftSimpleView()
        let hostingController = UIHostingController(rootView: swiftSimpleView)
        addChild(hostingController)
        hostingController.view.frame = view.bounds
        view.addSubview(hostingController.view)
        hostingController.didMove(toParent: self)
    }
    
    public override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        setInteractiveRecognizer()
    }
    
    private func setInteractiveRecognizer() {
        guard let controller = navigationController else { return }
        popRecognizer = InteractivePopRecognizer(controller: controller)
        // 将当前controller的委托赋值为新建的委托 直接赋值为nil会导致意外情况发生
        controller.interactivePopGestureRecognizer?.delegate = popRecognizer
        self.navigationController?.setNavigationBarHidden(true, animated: false)
    }
    
    public override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // 将当前controller的委托重置为原来的委托
        self.navigationController?.interactivePopGestureRecognizer?.delegate = self
        self.navigationController?.setNavigationBarHidden(false, animated: false)
    }
    
}
```