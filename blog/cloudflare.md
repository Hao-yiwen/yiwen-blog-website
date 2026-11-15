---
title: 使用 Cloudflare AI Gateway 代理 OpenAI 请求
authors: [yiwen]
tags: [Cloudflare, OpenAI, AI]
date: 2024-02-01
---

在开发 AI 应用时，直接从国内环境向 OpenAI 发起请求往往会遇到网络连接不稳定的问题。自建代理不仅需要额外的服务器运维成本，在稳定性和延迟上也常常难以保障。

<!-- truncate -->

## 解决方案：Cloudflare AI Gateway

Cloudflare 提供了 **AI Gateway** 服务。借助 Cloudflare 的全球边缘网络，通过 AI Gateway 转发 OpenAI 等大模型 API 请求，不仅接入简单，而且在连接稳定性、缓存控制、请求日志与限流监控方面都比自建反代省心得多。

## 参考文档

- [Cloudflare AI Gateway 官方文档](https://developers.cloudflare.com/ai-gateway/get-started/connecting-applications/)
