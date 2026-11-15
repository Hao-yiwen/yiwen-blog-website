---
title: Android集成RN
sidebar_label: Android集成RN
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android集成RN

今天尝试移动端和Js代码分离的方式并行开发移动端项目和RN项目，所以想将RN集成进Android中。但是文档给的都是标准的RN结构，所以并没有合适的文档告诉我该如何进行集成，于是在翻看了很多文档后我的得出了集成方案，并且今晚实际踩了一些坑，此文档用来记录Android集成RN的方法和问题。

## 方法

1. settings.gradle添加
```gradle
includeBuild("./rnDemo/node_modules/@react-native/gradle-plugin")
```

2. 项目级别build.gradle添加
```
buildscript {
    dependencies {
        classpath("com.facebook.react:react-native-gradle-plugin")
    }
}
```

3. 在app中添加
```
plugins {
    //...省略
    id("com.facebook.react")
}

// 这是核心
react {
    entryFile = file("../rnDemo/index.js")
    root = file("../rnDemo")
    reactNativeDir = file("../rnDemo/node_modules/react-native")
}

dependencies {
    //...省略
    // react Native
    implementation("com.facebook.react:react-android")
    implementation("com.facebook.react:hermes-android")
}
```

:::danger
这样配置以后可以发开了，但是没有自动link能力，在集成第三方库的时候需要自己手动集成。
:::

4. myReactNativeActivity
```java
package io.github.haoyiwen.react_native_container;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.KeyEvent;

import androidx.annotation.Nullable;

import com.facebook.react.BuildConfig;
import com.facebook.react.ReactInstanceManager;
import com.facebook.react.ReactPackage;
import com.facebook.react.ReactRootView;
import com.facebook.react.common.LifecycleState;
import com.facebook.react.modules.core.DefaultHardwareBackBtnHandler;
import com.facebook.react.shell.MainReactPackage;
import com.facebook.soloader.SoLoader;

import java.util.Arrays;


public class ReactNativeActivity extends Activity implements DefaultHardwareBackBtnHandler {

    private ReactRootView mReactRootView;

    private ReactInstanceManager mReactInstanceManager;

    private final int OVERLAY_PERMISSION_REQ_CODE = 1;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        /**
         * @description 为开发错误叠加层配置权限
         */
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(this)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivityForResult(intent, OVERLAY_PERMISSION_REQ_CODE);
            }
        }

        SoLoader.init(this, false);

        mReactRootView = new ReactRootView(this);

        mReactInstanceManager = ReactInstanceManager.builder()
                .setApplication(getApplication())
                .setCurrentActivity(this)
                .setBundleAssetName("index.android.bundle")
                .setJSMainModulePath("index")
                .addPackages(Arrays.<ReactPackage>asList(
                        new MainReactPackage()
                ))
                .setUseDeveloperSupport(BuildConfig.DEBUG)
                .setInitialLifecycleState(LifecycleState.RESUMED)
                .build();

        mReactRootView.startReactApplication(mReactInstanceManager, "MyReactNativeApp", null);

        setContentView(mReactRootView);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == OVERLAY_PERMISSION_REQ_CODE) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (!Settings.canDrawOverlays(this)) {
                    // SYSTEM_ALERT_WINDOW permission not granted
                }
            }
        }
        mReactInstanceManager.onActivityResult(this, requestCode, resultCode, data);
    }

    @Override
    public void invokeDefaultOnBackPressed() {
        super.onBackPressed();
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (mReactInstanceManager != null) {
            mReactInstanceManager.onHostPause(this);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (mReactInstanceManager != null) {
            mReactInstanceManager.onHostResume(this, this);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (mReactInstanceManager != null) {
            mReactInstanceManager.onHostDestroy(this);
        }

        if (mReactRootView != null) {
            mReactRootView.unmountReactApplication();
        }
    }

    @Override
    public void onBackPressed() {
        if (mReactInstanceManager != null) {
            mReactInstanceManager.onBackPressed();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_MENU && mReactInstanceManager != null) {
            mReactInstanceManager.showDevOptionsDialog();
            return true;
        }
        return super.onKeyUp(keyCode, event);
    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {
        super.onPointerCaptureChanged(hasCapture);
    }
}
```

5. index.js
```js
import React from 'react';
import {AppRegistry, StyleSheet, Text, View} from 'react-native';

const HelloWorld = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.hello}>Hello, World</Text>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  hello: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
});

AppRegistry.registerComponent(
  'MyReactNativeApp',
  () => HelloWorld,
);
```

## 常见问题

1. react模块必须配置到app中，在library中配置react模块是无效的（此问题查了1个多小时）