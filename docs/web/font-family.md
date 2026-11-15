---
title: Web 与移动端字体格式选型及推荐字体
sidebar_label: 字体格式与推荐
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Web 与移动端字体格式选型及推荐字体

## 一、主流字体格式对比

### 1. 跨端通用格式（Web & Native）

- **TrueType (`TTF`)**：Apple 与 Microsoft 联合推出的经典格式，操作系统与移动端原生支持极广，但体积偏大，未针对网络传输做专门压缩。
- **OpenType (`OTF`)**：基于 TTF 扩展（Adobe & Microsoft），支持更丰富的连字、复杂排版与大字符集，适合设计及客户端内嵌。

### 2. Web 专属优化格式

- **WOFF (`Web Open Font Format`)**：专为浏览器设计的压缩字体封装格式，体积显著小于 TTF/OTF，现代浏览器全面支持。
- **WOFF2**：采用 Brotli 算法深度压缩的下一代 Web 字体格式，体积比 WOFF 再小约 30%，加载极快，**现代 Web 首选**。
- **SVG Font**：早期基于矢量 XML 的字体方案，现已基本被图标字体/纯 SVG 图标与 WOFF2 取代。

## 二、实战踩坑：解决 Next.js 自定义字体首屏闪烁（FOIT/FOUT）

在开发网站首页引入外部自定义中文字体时，曾遇到首屏进入瞬间文字闪烁跳动的问题。起初仅配置了 `Next.js` 预加载仍未彻底改善，随后将原始 TTF 字体转换为高压缩率的 **`WOFF2`** 格式加载，传输耗时大幅下降，闪烁问题迎刃而解。

> **最佳实践**：Web 项目中优先提供 `woff2` 格式（必要时以 `woff` 兜底）；移动端 App 内置字体则使用精简子集化后的 `ttf` / `otf`。

## 三、免费商用字体推荐

- [站酷系列免费商用字体（Iconfont 字体库）](https://www.iconfont.cn/fonts)
