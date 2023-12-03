---
sidebar_position: 6
---

# node-gyp

`nodejs`在运行时候有时候需要`python`和`c++`编译环境，为什么会这样？

## 为什么需要node-gyp
 
为什么需要`node-gyp`,在`nodejs`中有一些库需要对原生侧进行操作和使用，例如使用和操作原生侧的线程之类。此时`nodejs`就会使用`node-gyp`来和原生交互，但是`node-gyp`是基于`python`编写的`gyp`的`nodejs`版本，所以在`node-gyp`运行的时候需要`python`环境。而在我们常用的系统`windows`和`macos`中，`c++`环境也是底层实现的主要方式，所以`node-gyp`需要`c++`编译环境，在`macos`中通常安装`xcode`解决，在`windows`中通常安装`visula studio`解决。