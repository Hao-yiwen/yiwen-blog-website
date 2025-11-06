---
title: 基于BaseActivity添加loading图标动画
sidebar_label: 基于BaseActivity添加loading图标动画
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 基于BaseActivity添加loading图标动画

动画使用lottie动画，效果更好。

1. BaseActivity

```java
import android.content.res.Configuration;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.airbnb.lottie.LottieAnimationView;

public abstract class BaseActivity extends AppCompatActivity {
    LottieAnimationView lottieAnimationView;
    LinearLayout loadingLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        int currentNightMode = getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK;
        if (currentNightMode == Configuration.UI_MODE_NIGHT_NO) {
            // Night mode is not active, we're using the light theme
            Log.d("BigHomeActivity", "onCreate: light theme");
            setTheme(R.style.AppTheme);
        } else if (currentNightMode == Configuration.UI_MODE_NIGHT_YES) {
            // Night mode is active, we're using dark theme
            Log.d("BigHomeActivity", "onCreate: dark theme");
            setTheme(R.style.AppTheme_Dark);
        }

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_base);

        LayoutInflater.from(this).inflate(getLayoutResId(), findViewById(R.id.container), true);

        lottieAnimationView = findViewById(R.id.lottieAnimationView);
        loadingLayout = findViewById(R.id.loadingLayout);
    }

    protected abstract int getLayoutResId();

    protected void showLoading() {
        if (lottieAnimationView != null) {
            loadingLayout.setVisibility(View.VISIBLE);
            lottieAnimationView.playAnimation();
        }
    }

    protected void hideLoading() {
        if (lottieAnimationView != null) {
            // 搞点动画小时时候的动画效果
            loadingLayout.animate()
                    .alpha(0f)
                    .setDuration(500)
                    .withEndAction(() -> {
                        loadingLayout.setVisibility(View.GONE);
                        lottieAnimationView.cancelAnimation();
                    })
                    .start();
        }
    }
}
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <FrameLayout
        android:id="@+id/container"
        android:layout_width="match_parent"
        android:layout_height="match_parent" />

    <LinearLayout
        android:id="@+id/loadingLayout"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        android:background="?attr/backgroundColor"
        android:gravity="center"
        android:orientation="vertical">

        <com.airbnb.lottie.LottieAnimationView
            android:id="@+id/lottieAnimationView"
            android:layout_width="200dp"
            android:layout_height="200dp"
            android:layout_centerInParent="true"
            android:visibility="visible"
            app:lottie_autoPlay="false"
            app:lottie_loop="true"
            app:lottie_rawRes="@raw/loading" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:gravity="center"
            android:text="加载中..."
            android:textColor="@color/black"
            android:textSize="16sp"
            android:visibility="visible" />
    </LinearLayout>

</FrameLayout>
```

2. 业务Activity

```java
import android.os.Bundle;
import android.os.Handler;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class BaseLoadingTestActivity extends BaseActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        showLoading();

        new Handler().postDelayed(() -> {
            hideLoading();
        }, 2000);
    }

    @Override
    protected int getLayoutResId() {
        return R.layout.activity_base_loading_test;
    }
}
```
