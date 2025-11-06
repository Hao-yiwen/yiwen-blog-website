---
title: viewBinding
sidebar_label: viewBinding
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# viewBinding

## 开启viewbinding

```gradle
buildFeatures {
    viewBinding = true
}
```

## xml示例如下

```xml
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/main"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:gravity="center"
    android:orientation="vertical"
    tools:context=".ViewBindingActivity">

    <TextView
        android:id="@+id/textView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="hello world" />

    <Button
        android:id="@+id/button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="点击" />

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:layout_marginTop="16dp"
        android:text="测试文字1" />

</LinearLayout>
```

## 生成viewbing类 并直接使用

```java
private ActivityViewBindingBinding binding;

@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    binding = ActivityViewBindingBinding.inflate(getLayoutInflater());
    setContentView(binding.getRoot());

    binding.button.setOnClickListener(v -> {
        binding.textView.setText("binding view");
    });

    Window window = getWindow();

    this.findViewById(R.id.button);
}
```

## 在fragment中使用
```java
binding = FragmentViewBindingBinding.inflate(inflater, container, false);
View view = binding.getRoot();
binding.linearLayout.setGravity(17);
ViewGroup.LayoutParams params = binding.linearLayout.getLayoutParams();
params.width = ViewGroup.LayoutParams.MATCH_PARENT;
binding.linearLayout.setLayoutParams(params);
binding.button.setOnClickListener(this);
return view;
```

## 总结

viewbinding是生成一个viewbinding的绑定类，直接使用绑定类的属性就可以，无需使用findviewbyId寻找了。
