---
title: 在interfacebuilder中添加UIScrollview
sidebar_label: 在interfacebuilder中添加UIScrollview
date: 2024-06-25
last_update:
  date: 2024-06-25
---

import uiscrollview from '@site/static/img/uiscrollview.png'

# 在interfacebuilder中添加UIScrollview

1. 添加`UIscrollview`到屏幕中
2. 添加`UIScrollview`的依赖，上下左右对其`safeview`
3. 在scrollview中添加view容器，然后view链接contentview的上下左右(还需要手动调整最trailing/bottom)以及水平居中在frameview中
4. 在view中添加任意内容
5. 注意：内容的bottom一定要设置和view的bottom的距离
6. 如果屏幕超出一屏，可手动调节显示屏幕的高度

<img src={uiscrollview} width={500} />