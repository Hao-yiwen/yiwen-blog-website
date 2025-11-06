---
title: 在Form中使用Upload组件
sidebar_label: 在Form中使用Upload组件
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 在Form中使用Upload组件

## 介绍

今天要做的一个功能是将本地图片上传至阿里云oss，这里没有采用只传阿里云的方案，而是通过后端中转。今天做的一个难点是在Antd的form中使用upload，如果图片存在则显示缩略图，图片不存在则显示上传按钮，限制是一张图片。[antd upload](https://ant.design/components/upload-cn#components-upload-demo-upload-manually)

## 示例代码

```tsx title="前端代码"
// 如果后端返回的url存在则需要再form中显示缩略图
if(record?.avatar) {
    setFileUrlList([
    {
        url: record.avatar,
        thumbUrl: record.avatar
    }
    ])
}
// 逻辑
const handleAdd = useCallback(async (values: any) => {
    const params = {...values};
    if(fileUrlList.length > 0) {
        params.avatar = fileUrlList[0].url; // 手动设置url
    }
    if(!isEdit) {
      const res = await request('/user/add', {
        method: 'post',
        data: params,
      });
      if (res.code === 1) {
        message.success('添加用户成功');
        await fetchDatas();
      }
    } else {
      const res = await request('/user/edit', {
        method: 'post',
        data: params,
      })
      if (res.code === 1) {
        message.success('修改用户信息成功');
        await fetchDatas();
      }
    }
    setVisible(false);
}, [isEdit, fileUrlList]);
// 组件
<Form.Item
    name="avatar"
    label="头像"
    getValueFromEvent={() => ''} // 选择受控组件做法，不走默认逻辑
    >
    <Upload
        fileList={fileUrlList} // 受控组件
        action="http://localhost:8080/file/upload" // url
        listType="picture-card"
        maxCount={1}
        headers={{
        authorization:
            'Bearer ' + 'token', // jwt验证
        }}
        onRemove={(file: any) => {
            setFileUrlList((fileUrlList) => {
                return fileUrlList.filter((item) => item.uid !== file.uid);
            });
        }}
        onChange={(fileData) => {
            const fileList: any[] = fileData.fileList.map(file => {
                let url = file.url;
                let thumbUrl = file.thumbUrl;
                let name = file.name;
                if (file.response) {
                    url = file.response.url; // 返回下载图片的链接
                    thumbUrl = file.response.thumbUrl; // 缩略图链接
                    name = file.response.name; //图片名称
                }
                return {
                    ...file,
                    url,
                    thumbUrl,
                    name,
                }
            });
            setFileUrlList(fileList);
        }}
    >
        <button style={{ border: 0, background: 'none' }} type="button">
        <PlusOutlined />
        <div style={{ marginTop: 8 }}>Upload</div>
        </button>
    </Upload>
</Form.Item>
```

```java title="后端代码"
// 上传实现
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
// 存入数据库实现...(省略)
```
