---
sidebar_position: 2
---

# 为什么要设置uses-material-design

uses-material-design: true 是在 Flutter 的 pubspec.yaml 文件中设置的，它的作用是包含 Material Design 图标库在你的应用中。这样你就可以在你的应用中使用 Icons 类来访问这些图标。

即使你的应用不使用 Material 主题，你仍然可以设置 uses-material-design: true 来使用 Material 图标。这个设置并不会强制你的应用使用 Material 主题，它只是包含了图标库。

例如，你可以在任何地方使用 Icon(Icons.favorite) 来显示一个心形图标，无论你的应用是否使用 Material 主题。
