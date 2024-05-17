# docker镜像仓库迁移

## 前置条件

[阿里云容器仓库](https://cr.console.aliyun.com/cn-shanghai/instance/repositories)

## 使用github action将镜像打包上传至阿里云容器仓库

```yaml
name: Docker Build and Push to Aliyun

on:
  push:
    tags:
      - 'v*'  # 当推送符合 'v*' 格式的标签时触发

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Code
      uses: actions/checkout@v2

    - name: Login to Aliyun Container Registry
      run: |
        echo "${{ secrets.ALIYUN_CONTAINER_REGISTRY_PASSWORD }}" | docker login --username ${{ secrets.ALIYUN_CONTAINER_REGISTRY_USERNAME }} --password-stdin registry.cn-shanghai.aliyuncs.com

    - name: Extract Git Tag for Version
      id: tag
      run: echo "::set-output name=version::$(git describe --tags --abbrev=0)"

    - name: Build Docker Image
      run: |
        docker build --platform linux/amd64 -t registry.cn-shanghai.aliyuncs.com/${{ secrets.ALIYUN_CONTAINER_REGISTRY_NAMESPACE }}/yiwen-blog-website:${{ steps.tag.outputs.version }} .

    - name: Push to Aliyun Container Registry
      run: |
        docker push registry.cn-shanghai.aliyuncs.com/${{ secrets.ALIYUN_CONTAINER_REGISTRY_NAMESPACE }}/yiwen-blog-website:${{ steps.tag.outputs.version }}
```

说明：

- 容器版本使用tag标签版本来动态定义，而不是写死
- 设置对应的秘钥，在代码变化的时候手动push到容器仓库