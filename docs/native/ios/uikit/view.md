---
sidebar_position: 2
---

# UIkit基本示例

```swift
import UIKit

class ViewController: UIViewController {

    // 视图控制器的视图被加载到内存后调用。这是设置初始状态的好地方。
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.

        let label = UILabel(frame: CGRect(x: 0, y: 0, width: 200, height: 21))
        label.center = CGPoint(x: 160, y: 285)
        label.textAlignment = .center
        label.text = "Hello, World!"
        self.view.addSubview(label)
    }
}

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    // 程序入口
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = ViewController()
        window?.makeKeyAndVisible()
        return true
    }
}
```
