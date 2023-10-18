---
sidebar_position: 2
---

# || 和 ?? 区别

使用 ||（逻辑或）:

当Device.safeAreaTop为任何假值（falsy values）时，如0、null、undefined、false、NaN、''（空字符串）等，top都会被赋值为0。
这意味着即使Device.safeAreaTop的值是0，top也会被赋值为0（这听起来可能没有问题，但在某些场景下这种行为可能不是你想要的）。
使用 ??（空值合并运算符 / Nullish coalescing operator）:

当Device.safeAreaTop为null或undefined时，top才会被赋值为0。
对于其他的假值，如0、false、NaN、''，top会被赋值为Device.safeAreaTop的实际值。
