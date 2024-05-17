# android集成rn第三方依赖

## 手动连接

示例： `react-native-webview`

1. settings.gradle.kts

```
include(":react-native-safe-area-context")
project(":react-native-safe-area-context").projectDir = file("./rnDemo/node_modules/react-native-safe-area-context/android")
```

2. build.gradle

```
implementation(project(":react-native-safe-area-context")){
      exclude(group: "com.facebook", module: "io.github.haoyiwen.react_native_container")
  }
```

3. `sync project...`

4. 添加package

```java
ReactInstanceManagerBuilder tmp = ReactInstanceManager.builder()
    .setApplication(getApplication())
    .setCurrentActivity(this)
    .setBundleAssetName("index.android.bundle")
    .addPackages(Arrays.<ReactPackage>asList(
            new MainReactPackage(),
            new RNCWebViewPackage(),
            new MyReactPackage()
//                        new RNScreensPackage()
    ))
    .setUseDeveloperSupport(BuildConfig.DEBUG)
    .setInitialLifecycleState(LifecycleState.RESUMED)
    .setJavaScriptExecutorFactory(new HermesExecutorFactory());
```

## 问题

1. 在集成react-native-screens或者react-safe-area-context的时候一直有如下报错，后面提给了官方，但是pr被关闭了

```bash
Type com.facebook.react.viewmanagers.RNCSafeAreaProviderManagerDelegate is defined multiple times:
/Users/haoyiwen/Documents/android/android-study/xml-and-compose-view-samples/react-native-container/build/.transforms/3d32361675af7f728702c4294d7321f2/transformed/bundleLibRuntimeToDirDebug/bundleLibRuntimeToDirDebug_dex/com/facebook/react/viewmanagers/RNCSafeAreaProviderManagerDelegate.dex,
/Users/haoyiwen/Documents/android/android-study/xml-and-compose-view-samples/rnDemo/node_modules/react-native-safe-area-context/android/build/.transforms/77918962839e12bbe7bc3949592f1741/transformed/bundleLibRuntimeToDirDebug/bundleLibRuntimeToDirDebug_dex/com/facebook/react/viewmanagers/RNCSafeAreaProviderManagerDelegate.dex
```

[手动连接](https://github.com/th3rdwave/react-native-safe-area-context/issues/490#issuecomment-2054037773) 2. 配置自动连接脚本一直显示`react-cli`找不到
