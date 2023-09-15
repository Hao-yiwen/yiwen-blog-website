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

### 图片上传controller侧入参和出参

```java
@RequestMapping(value = "/upload/coursefile", consumes = MediaType.MULTIPART_FORM_DATA)
public UploadFileResultDto upload(@RequestPart("filedata")MultipartFile filedata) {
    return null;
}
```

## Xmind图

因为前期学习并没有写博客和文档的习惯，所以之前的学习笔记就用下面的`xmind`图来代替

![java学习框架](../../../static/img/java_jiagou.png)
