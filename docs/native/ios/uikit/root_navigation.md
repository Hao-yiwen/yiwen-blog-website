# 从任何地方获取UIKit中navigationController的办法

## 从oc和swift获取navigationController的方法

```swift
if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
    let window = windowScene.windows.first,
    let rootViewController = window.rootViewController {
    if let navigationController = rootViewController as? UINavigationController {
        navigationController.popViewController(animated: true)
    } else {
        rootViewController.navigationController?.popViewController(animated: true)
    }
}
```

```objectivec
UIWindow *window = nil;
for (UIWindowScene *windowScene in [UIApplication sharedApplication].connectedScenes) {
    if (windowScene.activationState == UISceneActivationStateForegroundActive) {
        window = windowScene.windows.firstObject;
        break;
    }
}

UIViewController *rootViewController = window.rootViewController;

if ([rootViewController isKindOfClass:[UINavigationController class]]) {
    UINavigationController *rootNavigationController = (UINavigationController *)rootViewController;
    // 创建并设置 WKWebViewScreenController
    WKWebViewScreenController *webViewController = [[WKWebViewScreenController alloc] init];
    webViewController.urlString = url;
    
    // 使用 navigationController 进行页面跳转
    [rootNavigationController pushViewController:webViewController animated:YES];
} else if (rootViewController.navigationController) {
    // 如果 rootViewController 不是 UINavigationController，但有 navigationController
    UINavigationController *navigationController = rootViewController.navigationController;
    WKWebViewScreenController *webViewController = [[WKWebViewScreenController alloc] init];
    webViewController.urlString = url;
    
    [navigationController pushViewController:webViewController animated:YES];
} else {
    NSLog(@"Root view controller is not a navigation controller and has no navigation controller");
}
```