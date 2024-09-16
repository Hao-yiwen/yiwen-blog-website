# 在弹窗等非页面场景获取安全区域

flutter中很多场景需要用到底部弹窗等组件，但是底部弹窗等组件无法正常使用safearea等组件，因为弹窗的context和view的context并不一致，有一种方案是使用状态流全局缓存，但是该方案在页面非常复杂的情况下会导致页面维护成本急剧增加。

所以有什么方案是能在弹窗中直接获取安全距离的那。

```dart
import 'dart:math' as math;
// 在弹窗中获取当前页面的安全区域padding
// https://stackoverflow.com/questions/49737225/safearea-not-working-in-persistent-bottomsheet-in-flutter
final view = View.of(context);
final viewPadding = view.padding;
final mediaPadding = MediaQuery.paddingOf(context);
final viewTopPadding = viewPadding.top / view.devicePixelRatio;
final topPadding = math.max(viewTopPadding, mediaPadding.top);
```