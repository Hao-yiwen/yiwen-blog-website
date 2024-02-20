# RN缓存清理

```bash
// 删除所有的watchman监听对象
// 删除所有metro的临时对象
watchman watch-del-all && rm -rf $TMPDIR/metro-* && rm -rf $TMPDIR/haste-map-*
```