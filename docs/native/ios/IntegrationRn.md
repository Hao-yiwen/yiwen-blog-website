---
sidebar_position: 4
---

import ios_main1 from "@site/static/img/ios_main1.png";
import ios_main2 from "@site/static/img/ios_main2.png";
import ios_main3 from "@site/static/img/ios_main3.png";
import ios_main4 from "@site/static/img/ios_main4.png";

# ios项目集成ReactNative项目

## 参考文档

[Integration with Ios swift's Apps](https://reactnative.dev/docs/integration-with-existing-apps)

## 使用npm init创建一个项目

## 添加package.json

```json
{
    "name": "MyReactNativeApp",
    "version": "0.0.1",
    "private": true,
    "scripts": {
        "start": "yarn react-native start"
    }
}
```

## 添加ReactNative

```bash
yarn add react-native
```

## 根据提示信息添加react

```
yarn add react@version_printed_above
```

## 添加metro.config.js

```js
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://facebook.github.io/metro/docs/configuration
 *
 * @type {import('metro-config').MetroConfig}
 */
const config = {};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
```

## 将项目结构设置如下模样

<img src={ios_main4} style={{ width: 400 }} />

## 在ios文件夹创建项目

使用`Xcode`创建`storyboard`类型的`swift`项目。

## 删除Main storyboard相关配置

<img src={ios_main1} style={{ width: 600 }} />

<img src={ios_main2} style={{ width: 600 }} />

## 挂载metro打包后的产物

```swift
import UIKit
import React

@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let mockData:NSDictionary = ["scores":
          [
              ["name":"Alex", "value":"42"],
              ["name":"Joel", "value":"10"]
          ]
        ]

        let jsCodeLocation = URL(string: "http://127.0.0.1:8081/index.bundle?platform=ios")!
          let rootView = RCTRootView(
              bundleURL: jsCodeLocation,
              moduleName: "RNHighScores",
              initialProperties: mockData as [NSObject : AnyObject],
              launchOptions: nil
          )
          let vc = UIViewController()
          vc.view = rootView
        self.window = UIWindow(frame: UIScreen.main.bounds)
        window?.backgroundColor = .white
        window!.rootViewController = vc
        window!.makeKeyAndVisible()
        return true
    }
}
```

## pod init 初始化podfile

## 修改podfile

````ruby
# Resolve react_native_pods.rb with node to allow for hoisting
require Pod::Executable.execute_command('node', ['-p',
  'require.resolve(
    "react-native/scripts/react_native_pods.rb",
    {paths: [process.argv[1]]},
  )', __dir__]).strip

platform :ios, min_ios_version_supported
prepare_react_native_project!

# If you are using a `react-native-flipper` your iOS build will fail when `NO_FLIPPER=1` is set.
# because `react-native-flipper` depends on (FlipperKit,...) that will be excluded
#
# To fix this you can also exclude `react-native-flipper` using a `react-native.config.js`
# ```js
# module.exports = {
#   dependencies: {
#     ...(process.env.NO_FLIPPER ? { 'react-native-flipper': { platforms: { ios: null } } } : {}),
# ```
flipper_config = ENV['NO_FLIPPER'] == "1" ? FlipperConfiguration.disabled : FlipperConfiguration.enabled

linkage = ENV['USE_FRAMEWORKS']
if linkage != nil
  Pod::UI.puts "Configuring Pod with #{linkage}ally linked Frameworks".green
  use_frameworks! :linkage => linkage.to_sym
end

target 'rnDemo1' do
  config = use_native_modules!

  # Flags change depending on the env values.
  flags = get_default_flags()

  use_react_native!(
    :path => config[:reactNativePath],
    # Hermes is now enabled by default. Disable by setting this flag to false.
    :hermes_enabled => flags[:hermes_enabled],
    :fabric_enabled => flags[:fabric_enabled],
    # Enables Flipper.
    #
    # Note that if you have use_frameworks! enabled, Flipper will not work and
    # you should disable the next line.
    :flipper_configuration => flipper_config,
    # An absolute path to your application root.
    :app_path => "#{Pod::Config.instance.installation_root}/.."
  )

  post_install do |installer|
    # https://github.com/facebook/react-native/blob/main/packages/react-native/scripts/react_native_pods.rb#L197-L202
    react_native_post_install(
      installer,
      config[:reactNativePath],
      :mac_catalyst_enabled => false
    )
    # m系列mac必加
    __apply_Xcode_12_5_M1_post_install_workaround(installer)
  end
end
````

## 外层结构使用yarn安装依赖

## pod install安装依赖

## 修改Sandbox配置，否则运行时候会报错

<img src={ios_main3} style={{ width: 600 }} />

## 使用xcode打包壳工程

## 外层结构运行metro服务

```bash
react-native start
```
