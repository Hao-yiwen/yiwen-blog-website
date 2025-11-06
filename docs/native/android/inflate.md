---
title: inflate和replace
sidebar_label: inflate和replace
date: 2024-07-11
last_update:
  date: 2024-07-11
---

# inflate和replace

## 替换fragment（这是专门来替换fragment的api）

常用在activity中将占位view替换为fragment。

```java
FragmentManager fragmentManager = getSupportFragmentManager();
FragmentTransaction transaction = fragmentManager.beginTransaction();
transaction.replace(R.id.fragment_container, new YourFragment());
transaction.commit();
```

## inflate(膨胀)

```java
LayoutInflater inflater = LayoutInflater.from(context);
View view = inflater.inflate(R.layout.your_layout, container, false);
```

### 第三个参数解释

-   false： 只是返回膨胀的layout的根view，并不会讲R.layout.your_layout加载到container中。返回的是R.layout.your_layout的root view
-   true: 将R.layout.your_layout加载到container中，返回的是container。
