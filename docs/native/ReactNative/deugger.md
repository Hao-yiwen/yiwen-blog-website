# RN调试

## 背景

周六在开发的时候无意之间想要调试一下RN项目，然后就看官方文档调试，但是发现打断点死活不起作用，然后本着不信邪的原因坚持调试，本想着一定能够找到好的调试方案，但是经过一天的探索，发现断点依旧无法生效。

## 前置条件

- Hermes引擎
- 0.73.0/0.73.2

## vscode方案

- 下载react natvie tools调试工具
- 添加`attach to hermes application`
```json
{
    "name": "Attach to Hermes application - Experimental",
    "request": "attach",
    "type": "reactnativedirect",
    "cwd": "${workspaceFolder}/rnDemo0732",
    "port": 8083
},
```
- 有些时候断点可能不生效，不要慌心平气和解决问题~呜呜呜


## 描述

- 现在的情况是使用实验性调试方案的时候，刚进页面的断点无法触发，但是点击按钮后的断点还是能够正常触发的，非常离谱。vscode的调试方法到现在也没有跑通。

- flipper调试工具中`RN debugger Hermes`一直显示不可用，android和ios都一样。

- 查看dev中的产物代码，发现底部确实存在source url，所以source map应该没有问题，但是问题是为什么断点不起作用

- 产物url: `http://localhost:8081/index.bundle?platform=ios&dev=true`

- debugger调试也不起作用

## 最新进展 20240121

- 刚才无意之间意外的发现 如果按r重新刷新页面则无法获取到debugger 但是如果通过更改代码来触发热更新则可以进行调试 真是一个可怕的发现 一个有名的框架在稳定版本的调试系统竟然做的如此的垃圾 真是令人惊叹