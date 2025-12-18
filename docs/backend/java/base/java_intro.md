---
sidebar_position: 1
---

# Java

## 组成

- Java编程语言: Java作为一种编程语言，其核心是一套规则和语法，允许开发者编写源代码。Java语言以其简洁性、面向对象的特性、强类型系统、内存管理（垃圾回收）和跨平台能力（“编写一次，到处运行”）而闻名。

- Java虚拟机（JVM）: Java虚拟机是一个可以执行Java字节码的运行时环境。JVM是Java实现跨平台特性的关键，因为它允许相同的Java程序在不同的操作系统上运行，只要每个系统上都有对应的JVM实现。

- Java类库: Java类库是一个巨大的功能集合，包括了从基本数据结构到网络编程和图形用户界面（GUI）开发的所有内容。这些类库是Java平台的一部分，为Java程序提供了广泛的功能。

- Java Development Kit（JDK）: JDK是Java开发工具包，它提供了编译、运行Java程序所必需的工具，包括编译器（javac）、Java运行时环境（JRE），以及其他工具和类库。

- Java Runtime Environment（JRE）: JRE包括JVM、类库和其他运行Java程序所需的文件。它不包含编译器和开发工具，因此主要用于运行Java应用程序。

## QA

### 如何提取配置文件中的静态变量？

使用`springframework`的`config`配合`Value`来获取静态变量。

```java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MinioConfig {

    @Value("${minio.endpoint}")
    private String endpoint;

    @Value("${minio.accessKey}")
    private String accessKey;

    @Value("${minio.secretKey}")
   private String secretKey;
}
```

### 图片上传controller侧入参和出参？

```java
@RequestMapping(value = "/upload/coursefile", consumes = MediaType.MULTIPART_FORM_DATA)
public UploadFileResultDto upload(@RequestPart("filedata")MultipartFile filedata) {
    return null;
}
```

### e.printStackTrace()?

在 `Java` 中，`e.printStackTrace();` 用于打印异常（`Exception`）的堆栈轨迹到标准错误流（`stderr`）。这通常用于调试目的，以便了解异常是在哪里和如何被触发的。

当你调用 `e.printStackTrace();`，你会在控制台看到类似以下的输出：

```java
java.lang.NullPointerException
    at com.example.MyClass.someMethod(MyClass.java:16)
    at com.example.MyClass.anotherMethod(MyClass.java:10)
    at com.example.MyApp.main(MyApp.java:5)
```

这个输出会告诉你异常的类型（在这个例子中是`NullPointerException`）和异常是在哪里被触发的（哪个类、哪个方法、哪一行代码）。

然而，值得注意的是，在生产环境中通常不推荐使用 `e.printStackTrace();`，因为：

-   它输出到标准错误流，这可能会导致重要的错误信息被忽略或者与其他日志信息混合。
-   它可能会泄露敏感或者不必要的系统信息。
-   它是阻塞性的，这意味着它会阻塞当前线程直到堆栈轨迹被完全写出。

## Xmind图

因为前期学习并没有写博客和文档的习惯，所以之前的学习笔记就用下面的`xmind`图来代替

![java学习框架](../../../../static/img/java_jiagou.png)
