---
title: RN使用自定义字体
sidebar_label: RN使用自定义字体
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# RN使用自定义字体

1. 下载字体文件

从你选择的字体资源网站下载所需的字体文件（例如 .ttf 文件）。

2. 添加字体文件到项目中

将下载的字体文件添加到项目的 assets/fonts 目录中。如果这个目录不存在，可以创建它。

3. 配置 React Native 项目

在 React Native 项目的根目录下，创建或编辑 react-native.config.js 文件，添加字体的路径配置：

```
module.exports = {
  assets: ['./assets/fonts/'],
};
```

4. 链接字体

在添加字体和配置 react-native.config.js 文件后，重新构建项目：

```
npx react-native-asset
```

5. 使用自定义字体
   在项目中，通过 Text 组件使用自定义字体。在你的样式文件或组件样式中指定字体名称。例如：

```js
import React from 'react';
import { Text, StyleSheet, View } from 'react-native';

const App = () => {
    return (
        <View style={styles.container}>
            <Text style={styles.customFontText}>This is custom font text!</Text>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    customFontText: {
        fontFamily: 'YourCustomFontName', // 确保这里的字体名称与字体文件名匹配
        fontSize: 20,
    },
});

export default App;
```
