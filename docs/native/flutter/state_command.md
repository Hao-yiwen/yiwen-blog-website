---
title: 声明式UI和命令式UI
sidebar_label: 声明式UI和命令式UI
date: 2024-07-31
last_update:
  date: 2024-07-31
---

# 声明式UI和命令式UI

在目前的前端体系，应该会经常听到声明式UI和命令式UI两种UI写法。但是什么是声明式UI，而什么是命令式UI那，一直已来对这个概念非常不解。

但是在写flutter的过程汇总，看文章的时候忽然貌似顿悟了。大概理解了声明式UI和命令式UI的区别。

## 命令式UI

在早期的h5或者android还有ios开发中，都是开发者操作整个document、activity或者viewcontroller来进行布局的。例如我想讲某个组件设置为红色背景颜色，蓝色字体之类的。然后在交互或者各类IO的时候然后手动的更新到页面下一帧。这个过程是用户手动进行的，手动重写UI。
```java
Button btn = findViewId(R.id.btn);
TextView tv = findViewId(R.id.tv);
btn.setOnClickListener(v -> {
    tv.setBackgroundColor(R.color.red);
    // ....
})
```

以上差不多就是声明式UI的操作流程。

## 声明式UI

声明式UI的意思是用户只需要和View打交道，无需再操作整个视图和片段。也就是说框架本身会监听view，在view中触发各类IO的时候框架会自动管理视图。开发者只需要用声明来描述当前Ui就行，而没有了对整个视图的管理。

声明式UI由来依旧，但是重要里程碑是react框架。以下将通过react来分析当前声明式UI的逻辑。

react框架通过虚拟dom来维护dom树，然后给出对应的触发hook或者钩子。例如，开发者会声明很多很多组件，然后这些组件构成个当前整个视图，然后当触发IO操作的时候，开发者只需要设计好下一帧的Ui。然后使用例如setstate进行整个视图重构。在重构期间，框架会根据对新的虚拟tree和当前tree的比较，最小化diff更新视图。而无需开发者手动一个一个去触发。这样做极大的简化了开发者的开发效率，极大的简化了开发成本。

react根据自己的虚拟dom以及diff算法高效的实现了声明式UI框架，后来，因为整个前端是共通。android根据相同的思想推出了compose，ios推出了swiftui。虽然实现各不相同，但是这就是声明式UI的核心操作原理。通过框架多做事情，从而简化开发者的开发，因为开发者再也不用去关心哪里多更新了一个dom哪里少更新了一个dom。

声明式UI是整个大前端的发展方向，因为极大提高了开发效率。但是也可以通过描述瞅见声明式UI在底层做了大量的工作，从而使得开发者简化开发。所以声明式UI会提高开发者对底层理解成本，但是会提高开发效率。从react理解整个声明式UI是一个很好的选择。

```dart
TextStyle? _getTextStyle(BuildContext context) {
    if (!inCart) return null;

    return const TextStyle(
        color: Colors.black54,
        decoration: TextDecoration.lineThrough,
    );
}

@override
Widget build(BuildContext context) {
return ListTile(
    onTap: () {
    onCartChanged(product, inCart);
    },
    leading: CircleAvatar(
    backgroundColor: _getColor(context),
    child: Text(product.name[0]),
    ),
    title: Text(
    product.name,
    style: _getTextStyle(context),
    ),
);
}
```