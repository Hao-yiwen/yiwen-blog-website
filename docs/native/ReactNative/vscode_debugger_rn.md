---
title: 使用vscode调试rn
sidebar_label: 使用vscode调试rn
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import vscode_image from "@site/static/img/vscode_image.png";

# 使用vscode调试rn

## 背景

在开发rn项目的时候最让人头痛的一件事情就是调试，之前的方案一直是在chrome调试器中调试，因为rn中的断点在有些时候会漂移，所以尝尝用debugger来调试，但是用debugger来调试就有一个问题，需要再vscode中写debugger，而在chrome调试。那么问题来了，能否在vscode中既可以写代码也将调试一起进行了。经过对rn@0.74.0在vscode的调试发现，这一愿景完全是可行的。

## 步骤

1. 下载`react native tools`

2. 在调试部分选择rn的实验性调试器或者hermes调试

3. 插件会先创建一个`react native`包生成工具，这个工具相当于`npx react-native start`来启动服务

4. 然后根据设备选择任意调试工具，例如`attach to hermes application`

5. 此时就会出现vscode调试红条，也就是正式进入调试模式了。

<img src={vscode_image} width={700} />
