# Android添加c++

在 Android 应用程序中使用 C++ 代码通常通过 JNI（Java Native Interface）进行交互。JNI 提供了一种在 Java 和 C/C++ 代码之间调用方法和传递数据的机制。以下是一个详细的示例，展示如何在 Android 项目中使用 JNI 进行 Java 和 C++ 的交互。

1. 创建 CMakeLists.txt 文件

```txt
cmake_minimum_required(VERSION 3.4.1)

add_library(native-lib SHARED src/main/cpp/native-lib.cpp)

find_library(log-lib log)

target_link_libraries(native-lib ${log-lib})
```

2. 编写 C++ 代码
   在 app/src/main 目录下创建一个 cpp 文件夹，并在其中创建 native-lib.cpp 文件，内容如下：

```cpp
#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
    JNIEnv* env,
    jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
```

3. 编写 Java 代码
   在 MainActivity.java 中加载本地库并调用本地方法：

```java
package com.example.myapplication;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    // 加载 native-lib 库
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tv = findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
        // hello from jni
    }

    // 声明本地方法
    public native String stringFromJNI();
}
```

## c++代码解释

```cpp
#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
JNIEnv* env,
jobject /* this \*/) {
std::string hello = "Hello from C++";
return env->NewStringUTF(hello.c_str());
}
```

### 代码解释

1. 包含头文件

```
#include <jni.h>
#include <string>
```

-   #include `<jni.h>`: 引入 JNI（Java Native Interface）头文件，提供与 Java 交互的功能。
-   #include `<string>`: 引入标准库中的字符串类，用于处理 C++ 字符串。

2. 声明 C 函数调用约定

```cpp
extern "C" JNIEXPORT jstring JNICALL
```

-   extern "C": 告诉编译器使用 C 语言的命名约定来编译这段代码，以确保函数名在编译后的二进制文件中与 Java 期望的一致。
-   JNIEXPORT 和 JNICALL: 这些宏定义是 JNI 的一部分，确保函数可以被 Java 调用。

3. 定义 JNI 函数

```cpp
Java_com_example_myapplication_MainActivity_stringFromJNI(
    JNIEnv* env,
    jobject /* this */) {
```

-   Java*com_example_myapplication_MainActivity_stringFromJNI: 这是 JNI 函数的标准命名约定。格式为 Java*<包名>_<类名>_<方法名>，其中包名和类名中的点号替换为下划线。
-   JNIEnv\* env: 指向 JNI 环境的指针，用于与 JVM 交互。
-   jobject /_ this _/: 调用该本地方法的 Java 对象的引用。注释中的 this 表示这是一个未使用的参数。

4. C++ 字符串的定义和转换

```cpp
std::string hello = "Hello from C++";
return env->NewStringUTF(hello.c_str());
```

-   std::string hello = "Hello from C++";: 定义并初始化一个 C++ 字符串。
-   env->NewStringUTF(hello.c_str());: 使用 JNI 提供的 NewStringUTF 方法将 C++ 字符串转换为 Java 字符串，并返回给 Java 环境。hello.c_str() 返回 C++ 字符串的 C 风格字符串表示。

### JNI 相关说明

-   JNI 环境指针 (JNIEnv\*): 提供与 JVM 交互的方法，例如创建新字符串、调用 Java 方法等。
-   extern "C": 确保函数名在编译后的二进制文件中使用 C 语言命名约定，避免 C++ 名字修饰（name mangling）。
-   NewStringUTF: JNI 方法，用于创建新的 Java 字符串对象。
