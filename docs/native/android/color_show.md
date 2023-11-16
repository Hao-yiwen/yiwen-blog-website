---
sidebar_position: 2
---

# 颜色表示

## Web

-   十六进制（Hex）：如 #FFFFFF 表示白色。通常是6位（RGB），也可以是8位（ARGB）。

-   RGB：表示为 rgb(255, 255, 255)，分别代表红、绿、蓝色的强度。

-   RGBA：在 RGB 的基础上增加了 Alpha（透明度），如 rgba(255, 255, 255, 0.5)。

-   HSL：色相（Hue）、饱和度（Saturation）、亮度（Lightness），如 hsl(120, 100%, 50%)。

-   HSLA：HSL 加上 Alpha（透明度），如 hsla(120, 100%, 50%, 0.3)。

## Android

-   十六进制（Hex）：如 #RRGGBB 或 #AARRGGBB。例如，#FF0000 表示红色。(在颜色表示中，8位16进制（通常称为 ARGB 格式）是一种包含透明度信息的颜色编码方式。传统的6位16进制颜色代码（如 #FF5733）只包含红色、绿色和蓝色的信息，用于定义颜色的纯度和亮度。而8位16进制颜色代码在这基础上增加了两位用于表示透明度（alpha 值），格式为 #AARRGGBB。)
-   ARGB 函数：在 Java/Kotlin 代码中，可使用 Color.argb(int alpha, int red, int green, int blue)。
-   资源文件：在 colors.xml 中定义颜色，然后在布局文件或代码中引用。

## iOS

-   RGB(A) 函数：使用 UIColor(red:green:blue:alpha:) 方法。例如，UIColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0) 表示红色。
-   十六进制（需转换）：iOS 不直接支持 Hex，需要转换为 RGB。例如，#FF0000 需要转换为 UIColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0)。
-   预设颜色：如 UIColor.red。
    在 iOS 中，通常需要将十六进制颜色转换为 RGB 或使用系统提供的预设颜色值。在 Android 中，十六进制和资源文件是常用的颜色定义方式。
