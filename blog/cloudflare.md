# openai 代理

## 问题

最近在开发的时候遇到一个难题，就是墙的问题，有一个场景就是我把请求发送到 openai 的时候会被墙了，然后这个时候无需要使用魔法解决，但是魔法也不是万能的，也经常会被墙，所以这个时候如果有一个问题定的解决方案是非常好的，而自己维护的无论是代理还是魔法，必然在可靠性和性能侧有一些问题，那么如何解决这个问题那？

## 解决

-   cloudflare

cloudflare最近提供了一个AIgateway的服务，因为cloudflare在国内是有服务的，所以使用AIgateway能够做到高性能访问openai。比自己维护的无论在稳定性和性能侧面要好的多，真是一个宝藏选择。

## 文档

[AI gateway](https://developers.cloudflare.com/ai-gateway/get-started/connecting-applications/)
