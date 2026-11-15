---
title: 引用子模块的 Storyboard
sidebar_label: 引用子模块 Storyboard
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import substoryboard_controller from '@site/static/img/substoryboard_controller.png'
import storyboard_copy_bundle from '@site/static/img/storyboard_copy_bundle.png'

# 引用子模块的 Storyboard

当部分 Storyboard 存放在独立的 CocoaPods 子模块中，主工程通过 `Storyboard Reference` 跨模块引用时，常会出现找不到 ViewController 或 Bundle 资源未拷贝报错。

## 关键检查步骤

1. **Storyboard ID 与 Class 保持一致**：确保对应 ViewController 在 Identity Inspector 中配置了正确的 `Storyboard ID` 与 `Module`。

<img src={substoryboard_controller} width={500} />

2. **检查 Copy Bundle Resources**：确保子模块 `.podspec` 或主工程 `Build Phases -> Copy Bundle Resources` 中已包含对应的 `.storyboard` 资源包。

<img src={storyboard_copy_bundle} width={600} />
