---
title: IOS手动打包
sidebar_label: IOS手动打包
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# IOS手动打包

## 1. 创建开发者证书和配置文件

### 创建开发者证书

1.	登录 Apple Developer Portal。
2.	导航到 “Certificates, IDs & Profiles”。
3.	在左侧菜单中选择 “Certificates”，然后点击右上角的加号（+）按钮。
4.	选择 “iOS App Development” 或 “App Store and Ad Hoc”（根据需要选择），然后点击 “Continue”。
5.	按照提示创建证书请求（CSR）文件。可以通过 “Keychain Access” 应用程序在Mac上创建CSR文件：
-	打开 “Keychain Access” 应用程序。
-	在菜单栏中选择 “Keychain Access” -> “Certificate Assistant” -> “Request a Certificate from a Certificate Authority”。
-	输入你的电子邮件地址和常用名称，选择 “Saved to disk”，然后点击 “Continue”。
6.	上传CSR文件到Apple Developer Portal，然后下载生成的证书并双击安装到 “Keychain Access”。

### 创建App ID

1.	在 “Certificates, IDs & Profiles” 页面，选择 “Identifiers”。
2.	点击右上角的加号（+）按钮，选择 “App IDs”。
3.	输入描述名称和Bundle ID（例如com.yourcompany.yourapp），点击 “Continue”，然后点击 “Register”。

### 创建配置文件（Provisioning Profile）

1.	在 “Certificates, IDs & Profiles” 页面，选择 “Profiles”。
2.	点击右上角的加号（+）按钮，选择 “iOS App Development” 或 “App Store”（根据需要选择），然后点击 “Continue”。
3.	选择你之前创建的App ID，然后点击 “Continue”。
4.	选择你要使用的开发者证书，然后点击 “Continue”。
5.	选择要包括的设备（仅限开发配置文件），然后点击 “Continue”。
6.	输入配置文件名称，点击 “Generate”，然后下载并安装生成的配置文件。

## 2. 配置Xcode项目

1. 打开Xcode并加载你的项目。
2. 选择你的项目文件，在左侧的项目导航中选择你的目标（target）。
3. 在 “Signing & Capabilities” 选项卡中：
    - 勾选 “Automatically manage signing”（自动管理签名），或者取消勾选并手动选择你的团队（Team）。
    - 选择你之前创建的配置文件（Provisioning Profile）。
    - 确保选择了正确的证书（Certificate）。

## 3. 打包应用程序

### 在Xcode中打包

1.	选择你的设备或模拟器，确保设置为 “Any iOS Device (arm64)”。
2.	在菜单栏中选择 “Product” -> “Archive”。Xcode将构建并归档你的应用程序。
3.	构建完成后，Xcode将打开 “Organizer” 窗口。在这里你可以看到你创建的所有归档文件。
4.	选择你刚刚创建的归档文件，然后点击 “Distribute App”。
5.	选择 “Ad Hoc” 或 “App Store”（根据需要选择），然后点击 “Next”。
6.	选择你之前创建的配置文件，然后点击 “Export”。Xcode将生成一个IPA文件。

## 4. 手动签名

如果需要手动签名IPA文件，可以使用 codesign 和 xcrun 工具。

### 手动签名步骤

1.	生成IPA文件：
-	使用Xcode创建一个未签名的IPA文件。
2.	使用 codesign 对应用程序进行签名：

```bash
codesign -f -s "iPhone Distribution: Your Company (TEAMID)" --entitlements entitlements.plist Payload/YourApp.app
```

3. 使用 xcrun 重新打包IPA文件：

```bash
xcrun -sdk iphoneos PackageApplication -v Payload/YourApp.app -o ~/Desktop/YourApp.ipa
```
