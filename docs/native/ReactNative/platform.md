---
sidebar_position: 2
---

# Platform

```js
import { Platform } from 'react-native';

if (Platform.Version === 25) {
    console.log('Running on Nougat!');
}
```

-   在Android上，Platform.Version返回的是安卓设备的API级别。例如，API级别25对应于Android 7.1（Nougat）。[android版本查询](https://en.wikipedia.org/wiki/Android_version_history#Overview11)

-   在iOS上，它返回的是iOS设备的操作系统版本号，例如"10.3”。
