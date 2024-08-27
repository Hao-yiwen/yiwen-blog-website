---
sidebar_position: 1
---

# Flutter开发

## 学习文档

[flutter英文文档](https://docs.flutter.dev)

## 示例应用

[flutter和RN比较](https://github.com/Hao-yiwen/flutter_study/tree/master/flutter_application)

## 感想

flutter本地也是react声明式思想发展的产物，只不过flutter在google工程师的实现下，完成了从渲染到表面架构的完全重构，从而使得框架的实用性极大的增强。

-   js/dart层: dart层和react-native层中的js侧类似，都市使用虚拟树来进行声明式语法构建。只不过flutter对各种属性进行了拆分，从而使得语法树需要element tree中间树进行转换，从而比react-native的虚拟树多一层。
-   render层: 这一层就是设计上的完全不同，flutter使用自家skia引擎完成了渲染一条龙服务。而rn则是使用原生组件渲染。

flutter最终产物经过aot打成了so或者a文件，而rn则使用js或者优化后的hbc格式，所以性能差别巨大。其他平台适配性，flutter有embda层用来做平台移植，所以成本要低得多。而rn则是需要根据rn的组件规范重写底层，成本极高。

但是两个框架各有优点，并不是flutter完胜rn，因为对平台app来说，动态化的重要性或许要大于性能。最好的实现则是两者一起搭配使用，但是这样需要两份基建，对小型企业来说或许不太现实。

说了很多flutter优点，我尝试这用flutter来做项目开发，但是不知道是因为经验原因还是为什么，总是感觉flutter的开发成本要高于RN，在基建完善的情况下，RN的开发效率感觉是flutter的1.5倍左右，而RN的维护成本却也高于flutter。

**从长远来说，flutter一定是主流，而rn的发展方向也一定是自渲染，从yoga3从native侧迁移到c++层可知一二。**

~~在学习了`flutter`后，感觉`flutter`设计确实很现代化，很多地方和`rn`类似，但是在底层原理侧有着天绕之别。回想过去一个月的`swift`和`swiftui`学习过程，再和现在的`flutter`学习过程对比，真的体验差别好大，以为`flutter`框架的成熟以及规模化使用，中文文档也很完善，很多解决方案都有中文版本，这给母语不是英语的开发者提供了很大的便捷，~~
