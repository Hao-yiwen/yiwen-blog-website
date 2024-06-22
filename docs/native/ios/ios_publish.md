import fastlane from "@site/static/img/fastlane.png";

# IOS打包

## 前言

在经过20240622一晚上和20240623整整一天半夜的研究后，终于第一份使用fastlane自动化的github CI/CD能正常工作了。
该流水线能完成app打包，自动签名。并生成二维码来下载静态资源。

-  [仓库链接](https://github.com/Hao-yiwen/ios-demo1)

## 常规IOS打包

[文档](./manually_build.md)

如果看了手动打包的流程，我想大家可能就不行进行ios开发了，因为真的太繁琐太繁琐了。

而且最关键还不是繁琐的问题，如果后续我要添加测试设备，我必须要要重来一遍。重新创建配置文件吗，重新配置xcode。

这一套流程少则2日，多则1周。真的要搞死个人。所以人生苦短，我用fastlane (相比与android打包，fastlane还是复杂，但是相比于手动打包，那么这个就好的多。)

## 使用Fastlan打包

### 1. 安装fastLane

```bash
# 1. 在gem文件添加
gem "fastlane"
# 2.安装
bundle install
# 3.初始化fastlane
fastlane init
```

### 2. 使用fastlane match管理证书

#### 2.2 创建一个git仓库来存放证书

#### 2.2 初始化match

```bash
fastlane match init
```

#### 2.3 配置Matchfile

```bash
git_url("/your/cer/git/path")
storage_mode("git")
type("appstore")
api_key_path("./api-key.json")
app_identifier(["com.yiwen.ios-demo1"])
username("1552975589@qq.com")
```

```json
{
    "key_id": "app connect key",
    "issuer_id": "app connect issuer_id",
    "key": "base64化的app connect p8key",
    "is_key_content_base64": true, // 表明key是base64化的 否则会自动base64
    "in_house": false // 个人false 商业化是true
}
```

##### 参数解释

[前三个key的获取连接](https://appstoreconnect.apple.com/access/integrations/api)

1. key_id

-   描述: 这是 App Store Connect API 密钥的唯一标识符。每个密钥都有一个唯一的 key_id。
-   获取方式: 你可以在 App Store Connect 的用户和访问部分生成新的 API 密钥时获得这个 key_id。

2. issuer_id

-   描述: 这是一个用于标识 App Store Connect API 的发行者的唯一标识符。这个 issuer_id 是与你的 Apple 开发者帐户相关联的。
-   获取方式: 你可以在 App Store Connect 的 API 密钥页面找到这个 issuer_id。它通常在页面顶部列出。

3. key

-   描述: 这是你的 App Store Connect API 私钥的内容，通常是一个 .p8 文件的内容。你需要将这个私钥内容进行 base64 编码，以便在配置文件中使用。
-   获取方式: 当你生成新的 API 密钥时，Apple 会提供一个 .p8 文件，你需要保存这个文件的内容。

[应用专用密码获取连接](https://appleid.apple.com/account/manage)

4. 应用专用密码 (App-Specific Password)

-   描述: 应用专用密码是你 Apple ID 的一部分，用于在安全的环境中（如 CI/CD 系统）进行身份验证。它是一个用于访问 Apple 账户的辅助密码，通常用于与 Fastlane 等工具集成。
-   获取方式:
    1. 前往 Apple ID 账户页面。
    2. 登录并找到“安全”部分。
    3. 在“应用专用密码”部分，点击“生成密码”。
    4. 输入一个标签（如“Fastlane”）以标识这个密码，然后点击“创建”。
    5. 复制生成的密码并在 Fastlane 配置中使用。

#### 2.3 执行以下命令来生成需要的证书

```bash
# 发布到appstore
fastlane match appstore

# 开发使用证书
fastlane match development

# testflight使用证书
fastlane match adhoc
```

#### 2.4 取消自动签名

-   为什么取消自动签名？
    因为希望在github workflows自动打包，如果是自动签名则无法完成这一操作。所以使用手动注册的证书。证书有数量限制，在不是用的时候及时删除。


### 3. 配置fastlane
```bash
default_platform(:ios)

platform :ios do
  desc "Push a new beta build to TestFlight"
  lane :build_release do
    setup_ci if ENV['CI']
    match(type: "appstore", readonly: true)
    increment_build_number(xcodeproj: "ios-demo1.xcodeproj")
    build_app(
      scheme: "ios-demo1",
      export_method: "app-store",
      project: "ios-demo1.xcodeproj", # 使用project参数
      xcargs: "-allowProvisioningUpdates" # 允许Xcode自动处理配置文件
    )
    upload_to_testflight(
      # api_key_path: ENV['API_KEY_PATH']
      api_key_path: "./ios-study-api-key.json"
    )
  end

  desc "increment build version"
  lane :increment_version do
    increment_build_number(xcodeproj: "ios-demo1.xcodeproj")
  end

  desc "Build and sign a Debug package"
  lane :build_debug do
    setup_ci if ENV['CI']
    match(type: "development", readonly: true)
    build_app(
      scheme: "ios-demo1",
      export_method: "development",
      project: "ios-demo1.xcodeproj", # 使用project参数
      configuration: "Debug",
      export_options: {
        provisioningProfiles: {
          "com.yiwen.ios-demo1" => "match Development com.yiwen.ios-demo1"
        },
        generate_dsym: true, # 确保生成dSYM文件
        include_symbols: true, # 包括符号
        compileBitcode: false, # 根据需要设置，如果你不需要bitcode，可以设置为false
      },
      output_directory: "./build",
      output_name: "ios-demo1-debug.ipa"
    )
  end
end
```

以上为三个常用的lane:
- 一个是打出release包，上传到testflight
- 打出debug包
- 增加构建号的脚本

### 4.在终端执行以上脚本
```bash
fastlane build_debug
fastlane build_release
fastlane increment_version
```

### 5.使用github CI/CD自动化

em~ 如果生成二维码就更好了，所以yml它来了。

```yml tutle="build_and_deploy_testflight"
name: build_release_and_deploy_testflight

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: macos-latest
    permissions:
      contents: write  # 确保 GITHUB_TOKEN 有足够的权限

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up SSH
        uses: webfactory/ssh-agent@v0.5.3
        with:
          ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}

      - name: Set up Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: 3.3.3

      - name: Install dependencies
        run: |
          gem install bundler
          bundle install
          gem install fastlane

      - name: Decode and create API key file
        run: echo ${{ secrets.BASE64_API_KEY_CONTENT }} | base64 --decode > ios-study-api-key.json

      - name: Run Fastlane build_release lane
        run: fastlane build_release
        env:
          API_KEY_PATH: ios-study-api-key.json
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          FASTLANE_USER: ${{ secrets.FASTLANE_USER }}
          FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD: ${{ secrets.FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD }}
          KEY_ID: ${{ secrets.KEY_ID }}
          ISSUER_ID: ${{ secrets.ISSUER_ID }}

      - name: Run Fastlane build_debug lane
        run: fastlane build_debug
        env:
          API_KEY_PATH: ios-study-api-key.json
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          FASTLANE_USER: ${{ secrets.FASTLANE_USER }}
          FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD: ${{ secrets.FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD }}
          KEY_ID: ${{ secrets.KEY_ID }}
          ISSUER_ID: ${{ secrets.ISSUER_ID }}
      
      - name: Install qrencode
        run: brew install qrencode
  
      - name: Generate QR code for ios-demo1-debug.ipa
        run: |
          qrencode -o ios-demo1-debug-qr.png "https://github.com/${{ github.repository }}/releases/download/${{ github.ref }}/ios-demo1-debug.ipa"
          qrencode -o ios-demo1-qr.png "https://github.com/${{ github.repository }}/releases/download/${{ github.ref }}/ios-demo1.ipa"

      - name: Create Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body: |
            Release notes for ${{ github.ref_name }}
            Debug IPA QR Code: 
            ![Debug IPA QR Code](https://github.com/${{ github.repository }}/releases/download/${{ github.ref_name }}/ios-demo1-debug-qr.png)
            Release IPA QR Code:
            ![Release IPA QR Code](https://github.com/${{ github.repository }}/releases/download/${{ github.ref_name }}/ios-demo1-qr.png)
          draft: false
          prerelease: false
  
      - name: Upload IPA to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: /Users/runner/work/ios-demo1/ios-demo1/ios-demo1.ipa
          asset_name: ios-demo1.ipa
          asset_content_type: application/octet-stream

      - name: Upload IPA to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: /Users/runner/work/ios-demo1/ios-demo1/build/ios-demo1-debug.ipa
          asset_name: ios-demo1-debug.ipa
          asset_content_type: application/octet-stream

      - name: Upload dSYM to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: /Users/runner/work/ios-demo1/ios-demo1/ios-demo1.app.dSYM.zip
          asset_name: ios-demo1.dSYM.zip
          asset_content_type: application/zip
      
      - name: Upload dSYM to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: /Users/runner/work/ios-demo1/ios-demo1/build/ios-demo1-debug.app.dSYM.zip
          asset_name: ios-demo1-debug.dSYM.zip
          asset_content_type: application/zip
      
      - name: Upload QR codes to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./ios-demo1-debug-qr.png
          asset_name: ios-demo1-debug-qr.png
          asset_content_type: image/png

      - name: Upload QR codes to Release
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.PAT_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./ios-demo1-qr.png
          asset_name: ios-demo1-qr.png
          asset_content_type: image/png
```

这一百多行的配置文件需要配合以下环境变量使用

<img src={fastlane} width={500} />

- BASE64_API_KEY_CONTENT: base64化后的app-key-id.json
- FASTLANE_APPLE_APPLICATION_SPECIFIC_PASSWORD: 应用专用密码
- FASTLANE_USER: appleid
- ISSUER_ID: ISSUER_ID
- KEY_ID: KEY_ID
- MATCH_PASSWORD: match时候加密和解密的密码
- PAT_TOKEN: githubtoken,以为要写资源，所以需要添加write权限
- SSH_PRIVATE_KEY: 能够clone git@xxx仓库的ssh权限