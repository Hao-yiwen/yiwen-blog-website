---
title: IDEA中的搜索
sidebar_label: IDEA中的搜索
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# IDEA中的搜索

IntelliJ IDEA中的双击Shift和Shift + Command + F（在Windows/Linux上是Shift + Ctrl + F）提供了两种不同的搜索功能，它们分别适用于不同的搜索场景：

## 双击 Shift - Search Everywhere

双击Shift是IntelliJ IDEA的Search Everywhere功能。它是一个强大的搜索工具，允许你在一个统一的界面中搜索几乎所有事物，包括但不限于：

-   类和文件名
-   工具窗口和设置选项
-   动作和设置
-   符号（例如方法和变量名）

Search Everywhere的目的是提供一个快速入口，让你无需精确指定搜索类型就能找到几乎任何东西。它可以视作IDE的全局搜索入口，但它的搜索范围主要是项目文件、IDE的设置和动作等，而不直接针对代码内容的全文搜索。

## Shift + Command + F - Find in Files

Shift + Command + F（在Windows/Linux上是Shift + Ctrl + F）是Find in Files功能。这是一个专门用于在项目的所有文件中执行全文搜索的功能，它允许你根据具体的文本内容来搜索项目中的文件，包括：

-   在项目的所有文件中搜索指定的字符串或正则表达式。
-   限定搜索范围（例如，仅在特定目录或模块中搜索）。
-   指定更多搜索选项（如区分大小写、仅在评论中搜索等）。

Find in Files的主要用途是进行代码或资源文件的内容搜索，非常适合当你需要查找项目中使用了某个特定方法名、变量名或任何文本字符串的所有地方。

### 全局搜索中project module scope directory区别

- 选择 Project 适用于跨整个项目的广泛搜索。
- 选择 Module 当你需要在特定模块内搜索时。
- 选择 Scope 适用于需要在非常具体的文件集合中搜索时。
- 选择 Directory 适用于当你知道所查内容大致在哪个目录下时。