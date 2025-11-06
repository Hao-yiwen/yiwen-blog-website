---
title: Build Phase
sidebar_label: Build Phase
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Build Phase

在 Xcode 中，Build Phases 定义了在项目构建过程中的各个阶段应该执行的任务。这些任务包括编译源代码、复制资源文件、链接库文件等。Build Phases 在整个构建过程中有特定的执行顺序，既包括在构建之前，也包括在构建之后执行的任务。

## Build Phases 的顺序和类型：
- Pre-actions（构建之前）: 在 Build Phases 的界面中不直接显示，但可以在 Xcode 的 scheme 设置中定义。这些动作在构建过程开始之前执行，通常用于设置环境变量或执行脚本。

- Target Dependencies（目标依赖）: 确定当前构建目标（target）依赖于项目中的其他目标。如果有依赖关系，那么这些依赖目标会先被构建。

- Compile Sources（编译源码）: 编译项目中的源代码文件。这是构建过程的核心步骤之一。

- Link Binary With Libraries（链接二进制与库）: 将编译后的代码与所需的库文件（包括系统库和第三方库）链接起来。

- Copy Bundle Resources（复制捆绑资源）: 将图片、音频、xib文件等资源复制到构建的应用程序包中。

- Run Script（运行脚本）: 在这个阶段，你可以添加自定义脚本来执行特定的任务，如代码签名、修改资源文件等。这些脚本可以配置为在编译前或编译后执行。

- Post-actions（构建之后）: 和 Pre-actions 类似，这些也是在 scheme 设置中定义的动作，它们在整个构建过程完成后执行。

## 总结：
- Build Phases 包括了在构建过程的不同阶段执行的任务，它们按照特定的顺序进行。
- Run Script 阶段特别灵活，开发者可以配置脚本在构建之前或构建之后执行，取决于具体的需求。
- 通过合理配置和使用 Build Phases，可以有效地控制和自动化构建过程，优化开发流程