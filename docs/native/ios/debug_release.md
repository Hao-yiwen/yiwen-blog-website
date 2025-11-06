---
title: 不同模式打出不同包
sidebar_label: 不同模式打出不同包
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 不同模式打出不同包

在实际开发中，以为生产和开发往往又不用的配置，例如debug环境有详细的日志输出以及自建的debug配置项，以及网络环境。

那么在开发过程中打出两个包就是更好的选择，这样就能根据需要去进行开发调试了，在release环境看性能，在debug环境进行软件开发。

以下是如何在ios中打出debug和release包。

## 步骤

1. 在`build settings`搜索`prodeuct bundle identifier`,为debug和release环境设置不同identifier。例如后面加.debug和.release
2. 更改`general`中的`identity`：release保持不变，debug更改为`xxx-debug`

## 排障

```bash
Embedded binary's bundle identifier is not prefixed with the parent app's bundle identifier.
```

-   原因：更改了app的identify，extension也需要相应的更改。

-   解决方案：

如果原来app叫做`xxx`,widget叫做`xxx.widget`

那么现在app改为了`xxx.debug`,那么widget需要改为`xxx.debug.widget`

:::danger
注意不是`xxx.debug.widget`，不然就会一直有这个问题。
:::
