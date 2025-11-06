---
title: 上传文件至阿里OSS
sidebar_label: 上传文件至阿里OSS
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 上传文件至阿里OSS

使用阿里云OSS作为静态资源保存场所。[使用文档](https://help.aliyun.com/zh/oss/getting-started/sdk-quick-start?spm=a2c4g.11186623.0.0.2d4a4a74iqsgTm)

## 上传示例

### 依赖

```
implementation 'com.aliyun.oss:aliyun-sdk-oss:3.15.1'
implementation 'javax.xml.bind:jaxb-api:2.3.1'
implementation 'javax.activation:activation:1.1.1'
implementation 'org.glassfish.jaxb:jaxb-runtime:2.3.3'
```

### spring boot示例

```java
public FileDto upload(@RequestParam("file") MultipartFile file) {
    if (file.isEmpty()) {
        return null;
    }
    String fileName = FileUtil.generateUniqueFileName(file.getOriginalFilename());

    CredentialsProvider credentialsProvider = new DefaultCredentialProvider(accessKeyId, accessKeySecret);
    OSS ossClient = new OSSClientBuilder().build(endpoint, credentialsProvider);
    FileDto fileDto = new FileDto();
    try {
        ossClient.putObject(bucketName, fileName, file.getInputStream());
        fileDto.setThumbUrl("https://" + bucketName + "." + endpoint + "/" + fileName);
        fileDto.setUrl("https://" + bucketName + "." + endpoint + "/" + fileName);
        fileDto.setName(fileName);
        fileDto.setStatus("done");
        return fileDto;
    } catch (Exception e) {
        log.error("Error uploading file to OSS", e);
        fileDto.setStatus("error");

    }
    return fileDto;
}
```