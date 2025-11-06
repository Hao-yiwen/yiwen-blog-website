---
title: Android中xml的主题切换
sidebar_label: Android中xml的主题切换
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android中xml的主题切换

在 Android 中，通过使用 XML 文件和主题，可以轻松实现根据系统主题（白天模式和夜间模式）动态切换背景颜色。以下是详细的步骤：

1. 定义颜色资源
   首先，需要在 res/values/colors.xml 和 res/values-night/colors.xml 中定义颜色资源。

res/values/colors.xml (白天模式颜色)

```xml
<resources>
<color name="backgroundColorLight">#FFFFFF</color> <!-- 白色背景 -->
<color name="backgroundColorDark">#000000</color> <!-- 黑色背景 -->
</resources>
```

res/values-night/colors.xml (夜间模式颜色)

```xml
<resources>
<color name="backgroundColorLight">#000000</color> <!-- 黑色背景 -->
<color name="backgroundColorDark">#FFFFFF</color> <!-- 白色背景 -->
</resources>
```

2. 定义自定义属性
   在 res/values/attrs.xml 中定义自定义属性 backgroundColor。

res/values/attrs.xml

```xml
<resources>
<attr name="backgroundColor" format="reference|color"/>
</resources>
```

3. 定义主题
   在 res/values/themes.xml 和 res/values-night/themes.xml 中定义应用的主题，并使用自定义属性。

res/values/themes.xml (白天模式主题)

```xml
<resources xmlns:tools="http://schemas.android.com/tools">
<style name="AppTheme" parent="Theme.Material3.DayNight.NoActionBar">
<!-- 定义白天模式下的背景颜色 -->
<item name="backgroundColor">@color/backgroundColorLight</item>
</style>
</resources>
```

res/values-night/themes.xml (夜间模式主题)

```xml
<resources xmlns:tools="http://schemas.android.com/tools">
<style name="AppTheme" parent="Theme.Material3.DayNight.NoActionBar">
<!-- 定义夜间模式下的背景颜色 -->
<item name="backgroundColor">@color/backgroundColorDark</item>
</style>
</resources>
```

4. 使用自定义属性
   在布局文件中使用自定义属性来设置 LinearLayout 的背景颜色。

res/layout/activity_main.xml

```xml
<?xml version="1.0" encoding="utf-8"?>

<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center"
    android:background="?attr/backgroundColor"> <!-- 使用自定义属性设置背景颜色 -->

    <TextView
        android:id="@+id/textView"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Hello, World!"
        android:textSize="24sp"
        android:layout_marginTop="20dp" />

</LinearLayout>
```

5. 在 AndroidManifest.xml 中声明应用主题
   确保在 AndroidManifest.xml 文件中声明应用使用的主题。

AndroidManifest.xml

```xml
<application
    android:allowBackup="true"
    android:dataExtractionRules="@xml/data_extraction_rules"
    android:fullBackupContent="@xml/backup_rules"
    android:icon="@mipmap/ic_launcher"
    android:label="@string/app_name"
    android:roundIcon="@mipmap/ic_launcher_round"
    android:supportsRtl="true"
    android:theme="@style/AppTheme"
    tools:targetApi="31">
...
</application>
```

6. 确保活动继承正确的主题
   如果需要在 Activity 中动态切换主题，可以使用基类 BaseActivity 来处理主题变化，并让所有 Activity 继承该基类。

BaseActivity.java

```java
package com.example.javaviewtest;

import android.content.res.Configuration;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

public abstract class BaseActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // 根据系统的主题设置应用的主题
        int currentNightMode = getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK;
        if (currentNightMode == Configuration.UI_MODE_NIGHT_YES) {
            setTheme(R.style.AppTheme);
        } else {
            setTheme(R.style.AppTheme);
        }
        super.onCreate(savedInstanceState);
    }

    protected abstract int getLayoutResId();

}
```

7. 具体的 Activity
   所有的 Activity 都继承 BaseActivity，并实现 getLayoutResId 方法来提供布局资源 ID。

```java title="BigHomeActivity.java"
package com.example.javaviewtest;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.yiwen.compose_views.ComposeActivity;
import com.yiwen.recyclerviewtest.HomeActivity;
import io.flutter.embedding.android.FlutterActivity;

public class BigHomeActivity extends BaseActivity {

    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        showLoading();

        new Handler().postDelayed(() -> {
            hideLoading();
        }, 3000);

        TextView tv = findViewById(R.id.tv_title);
        tv.setText(stringFromJNI());

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        Toolbar toolbar = findViewById(R.id.toolbar);
        toolbar.setTitle("大首页");
        setSupportActionBar(toolbar);

        Button button = findViewById(R.id.btn_jump_home);
        button.setOnClickListener(v -> {
            Intent intent = new Intent(this, HomeActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        Button btn_compose = findViewById(R.id.btn_jump_compose);
        btn_compose.setOnClickListener(v -> {
            Intent intent = new Intent(this, ComposeActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        Button btn_jump_thrid_aar = findViewById(R.id.btn_jump_thrid_aar);
        btn_jump_thrid_aar.setOnClickListener(v -> {
            Intent intent = new Intent(this, com.example.chapter03.Chapter3BigHome.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        Button btn_jump_rn = findViewById(R.id.btn_jump_rn);
        btn_jump_rn.setOnClickListener(v -> {
            Intent intent = new Intent(this, io.github.haoyiwen.react_native_container.ReactNativeActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            intent.putExtra("initParam1", "value1");
            intent.putExtra("initParam2", 123);
            startActivity(intent);
        });

        Button btn_qr_code = findViewById(R.id.btn_jump_javaview_other);
        btn_qr_code.setOnClickListener(v -> {
            Intent intent = new Intent(this, com.yiwen.java_view_other.BigHomeActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        Button btn_jump_rn_fragment = findViewById(R.id.btn_jump_rn_fragment);
        btn_jump_rn_fragment.setOnClickListener(v -> {
            Intent intent = new Intent(this, com.example.javaviewtest.ReactNativeFragmentActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(intent);
        });

        Button btn_jump_flutter = findViewById(R.id.btn_jump_flutter);
        btn_jump_flutter.setOnClickListener(v -> {
            startActivity(
                    FlutterActivity.createDefaultIntent(this)
            );
        });
    }

    @Override
    protected int getLayoutResId() {
        return R.layout.activitu_constraint_layout_view1;
    }

    public native String stringFromJNI();

}
```

通过这些步骤，您可以根据系统主题自动切换应用的背景颜色。这种方式不仅可以保证应用在不同主题下的一致性，还可以减少代码重复。
