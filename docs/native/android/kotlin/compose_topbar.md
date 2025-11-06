---
title: compose设置Topbar颜色
sidebar_label: compose设置Topbar颜色
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# compose设置Topbar颜色

今天在开发compose应用的时候碰到一个问题，就是为什么给topbar设置顶部颜色的时候无法设置。

```kt
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DaysTopBar() {
    CenterAlignedTopAppBar(
        title = {
            Text(
                text = stringResource(id = R.string.app_name),
                style = MaterialTheme.typography.displayLarge
            )
        },
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer // 设置顶部应用栏的背景颜色
        )
    )
}
```

需要使用TopBar组件，然后在组件内部的colors属性中进行背景颜色配置，而不是在其中使用title，然后给title组件使用背景颜色