# ios侧pod慢的问题

- 在ios侧安装ios pod依赖，在国内有时候会非常慢。最慢的是boost和hermes-enigne。

- 在对ios相关知识进行学习了解后，逐渐知道了事情真正的原因。

- 国内网络对github clone速度限制非常严重，导致很多时候安装失败。

## 解决方案

在终端开加速器。

```bash
export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;export ALL_PROXY=socks5://127.0.0.1:1080
```

虽然开加速器后依然有几率被锁定，但是大概率安装会非常快。