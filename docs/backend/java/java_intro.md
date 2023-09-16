---
sidebar_position: 1
---

# Java

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

![java学习框架](../../../static/img/java_jiagou.png)
