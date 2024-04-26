import Size0740 from '@site/static/img/0740_size.png'
import Size0725 from '@site/static/img/0725_size.png'

# rn app大小

今天晚上尝试对一个新创建的`rn@0.74.0` android app打release包时发现，打完包后的体积是50mb，我心里咯噔了一下。这也太离谱了把，什么都没有做就50mb了。

## 0.74.0

<img src={Size0740} width={500}/>

:::danger
今天在学习android集成rn的时候无意发现，初始化一个0.74.0项目，直接打release包，体积竟然是50mb。我的天那~~
:::

## 0.72.5

<img src={Size0725} width={500}/>

## 总结

此事已提issues，[issues链接](https://github.com/facebook/react-native/issues/44291#issuecomment-2079889795)
