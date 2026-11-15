---
title: Xcode 构建常见问题（FAQ）
sidebar_label: FAQ
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Xcode 构建常见问题（FAQ）

## Xcode 15+ 报错 `Sandbox: rsync.samba deny(1) file-write-create`

### 报错日志

```text
Sandbox: rsync.samba(12698) deny(1) file-write-create .../DerivedData/.../Frameworks/Alamofire.framework/.Alamofire.ihLdr1
Sandbox: rsync.samba(12698) deny(1) file-write-create .../DerivedData/.../Frameworks/Alamofire.framework/.Info.plist.vn2fty
```

### 解决方案

Xcode 15 起默认开启了用户脚本沙盒限制，导致 CocoaPods 资源与 Framework 拷贝脚本被拦截。前往主 Target 的 **`Build Settings`** 搜索 **`User Script Sandboxing`**（即 `ENABLE_USER_SCRIPT_SANDBOXING`），将其设置为 **`No`** 即可解决。
