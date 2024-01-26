# RN性能测试

最近有个场景，需要测试一下`RN Harmony`和`RN android`的性能，在查略了很多资料后设计了一套测试`TTI/FCP`时间的方法。简单说就是，在原生侧`RN bundle`开始加载的时候记录时间，在`RN`侧最后一个元素完成渲染时的`onLayout`中打印加载完成时间，根据时间差来计算最终时间。

- 之前一直对`Android`如何打出`release`和`debugger`包存在困惑，现在搞清楚这个问题了。
- 经过测试，在元素较少的情景下`RN Harmony`和`RN android`相差不大(100个元素)
- 当元素上升到500个的时候，`RN Harmony`需要3s的加载时间
- 当元素上升到1500个的时候，`RN Harmony`需要18s的加载时间，而android只需要1s左右时间
