# ios和android集成flutter页面后调试

在混编app中，会遇到在宿主app中调试flutter页面的问题。

操作步骤也很简单。

## 步骤

1. 打出ios/android app的debug包

2. 用数据线连接android或者ios真机设备/使用模拟器

3. 在vscode打开`Dart: Attach to Process`调试命令，链接到flutter引擎，然后就可以开始热更新和断点调试了。

## 注意

1. ios经过测试需要先连接`debuging preocess`，然后再进行`Dart: Attach to Process`。否则会一直报`ios 14 xxx`报错。

## 疑问

以上都是个人开发体验，具体公司级别是怎么调试的需要看一下。
