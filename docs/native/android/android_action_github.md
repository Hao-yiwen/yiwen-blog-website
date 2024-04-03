# Android项目在github action中的打包

今天心血来潮，想要在github action中进行android打包，然后将打包产物上传到github 仓库页面的release中，经过查询和修补，这个是一个可用版本。

```yml
name: Android CI

on:
    push:
        tags:
            - 'v*' # 当推送符合 'v*' 格式的标签时触发

jobs:
    build:
        runs-on: ubuntu-latest

        steps:
            - uses: actions/checkout@v2

            - name: Set up JDK 17
              uses: actions/setup-java@v2
              with:
                  java-version: '17'
                  distribution: 'temurin'

            - name: Grant execute permission for gradlew
              run: chmod +x ./gradlew

            - name: Build Debug APK
              run: ./gradlew assembleDebug

            - name: Build Release APK
              run: ./gradlew assembleRelease

            - name: Extract Git Tag for Version
              id: tag
              run: echo "::set-output name=version::$(git describe --tags --abbrev=0)"

            - name: Create Release
              id: create_release
              uses: actions/create-release@v1
              env:
                  GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
              with:
              # 设置tag_name和release_name
                  tag_name: ${{ steps.tag.outputs.version }}
                  release_name: ${{ steps.tag.outputs.version }}
                  draft: false
                  prerelease: false

            - name: Upload APK to Release
              uses: actions/upload-release-asset@v1
              env:
                  GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
              with:
                  upload_url: ${{ steps.create_release.outputs.upload_url }}
                  asset_path: ./app/build/outputs/apk/debug/app-debug.apk
                  asset_name: app-debug.apk
                  asset_content_type: application/vnd.android.package-archive

            - name: Upload Release APK to Release
              uses: actions/upload-release-asset@v1
              env:
                  GITHUB_TOKEN: ${{ secrets.PERSONAL_ACCESS_TOKEN }}
              with:
                  upload_url: ${{ steps.create_release.outputs.upload_url }}
                  asset_path: ./app/build/outputs/apk/release/app-release.apk
                  asset_name: app-release.apk
                  asset_content_type: application/vnd.android.package-archive
```
