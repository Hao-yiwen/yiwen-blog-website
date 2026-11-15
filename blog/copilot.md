---
title: GitHub Copilot 使用笔记
authors: [yiwen]
tags: [GitHub Copilot, AI, 开发工具]
date: 2023-11-04
---

GitHub Copilot 是日常编码中极佳的 AI 结对编程提效工具，能够根据上下文与注释实时补全代码与单测。

<!-- truncate -->

## 支持的集成开发环境

- **VS Code**（官方插件 `GitHub Copilot` / `GitHub Copilot Chat`）
- **JetBrains 系列 IDE**（IntelliJ IDEA、WebStorm、PyCharm、Android Studio 等）
- **Neovim / Visual Studio / Xcode**

## 实用技巧

1. **注释驱动开发**：在函数上方先写出清晰的输入、输出与边界条件注释，Copilot 生成的实现准确度会大幅提升。
2. **快捷键熟练配合**：使用 `Tab` 采纳建议、`Alt/Option + ]` 与 `Alt/Option + [` 切换候选补全方案、`Cmd/Ctrl + →` 按词部分采纳。
3. **打开关联文件标签页**：Copilot 会参考当前编辑器已打开的相关类型定义与接口文件作为上下文，保持相关文件开启能显著减少幻觉。
