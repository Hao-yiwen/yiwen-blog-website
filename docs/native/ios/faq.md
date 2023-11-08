---
sidebar_position: 5
---

# FAQ

## QA1

```bash
Sandbox: rsync.samba(12698) deny(1) file-write-create /Users/shrek/Library/Developer/Xcode/DerivedData/PhotoPC-dejmlgrmwbxazrgvfwpxvhadwsuy/Build/Products/Debug-iphonesimulator/PhotoPC.app/Frameworks/Alamofire.framework/.Alamofire.ihLdr1

Sandbox: rsync.samba(12698) deny(1) file-write-create /Users/shrek/Library/Developer/Xcode/DerivedData/PhotoPC-dejmlgrmwbxazrgvfwpxvhadwsuy/Build/Products/Debug-iphonesimulator/PhotoPC.app/Frameworks/Alamofire.framework/.Info.plist.vn2fty
![]("https://developer.apple.com/forums/content/attachment/fb5c4e33-9603-4c87-9f39-aab81475dbf9" "title=Screenshot 2023-06-07 at 00.48.08.png;width=1429;height=232")
```

Check that `ENABLE_USER_SCRIPT_SANDBOXING` is disabled in the project's build settings.
