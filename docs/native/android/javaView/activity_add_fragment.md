---
title: Activity添加Fragment
sidebar_label: Activity添加Fragment
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Activity添加Fragment

今天在JavaView中添加添加ToolsBar,然后发现如果不进行封装，每一个页面度需要添加ToolsBar,所以写了一个Fragment来封装ToolsBar,也记录一下Activity添加Fragment.

## 声明Fragment.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    tools:context=".fragment.ToolBarFragment">

    <androidx.appcompat.widget.Toolbar
        android:id="@+id/toolbar_fragment"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:background="?attr/colorPrimary"
        android:elevation="4dp"
        android:theme="@style/ThemeOverlay.AppCompat.Dark.ActionBar" />

</FrameLayout>
```

## 编写Fragment.java

```java
public class ToolBarFragment extends Fragment {

    public ToolBarFragment() {
        // Required empty public constructor
    }

    public static ToolBarFragment newInstance(String param1) {
        ToolBarFragment fragment = new ToolBarFragment();
        Bundle args = new Bundle();
        args.putString("TITLE", param1);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_tool_bar, container, false);
        Toolbar toolbar = view.findViewById(R.id.toolbar_fragment);

        if (getArguments() != null && getArguments().containsKey("TITLE")) {
            String title = getArguments().getString("TITLE");
            toolbar.setTitle(title);
        }

        return view;
    }
}
```

## 在activity中使用

```xml
<FrameLayout
    android:id="@+id/fragment_toolbar"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    app:layout_constraintTop_toTopOf="parent" />
```

## 在Activity.java中添加

```xml
@Override
protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    EdgeToEdge.enable(this);
    setContentView(R.layout.activity_list_view);

    /**
        @description 添加fragment
    */
    if (savedInstanceState == null) {
        getSupportFragmentManager().beginTransaction().add(R.id.fragment_toolbar, ToolBarFragment.newInstance("ListView")).commit();
    }
}
```