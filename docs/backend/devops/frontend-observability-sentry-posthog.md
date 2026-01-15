---
title: 前端全链路监控方案
sidebar_position: 7
tags: [sentry, posthog, flutter, observability, apm, ab-test, analytics, frontend]
---

# 前端全链路监控方案：Sentry + PostHog

构建一套 **"端侧大一统监控体系"**，涵盖 **日志、Trace、Metric、APM、A/B Test、数据分析**。在开源界，最强悍的组合是 **Sentry + PostHog** 双核驱动。

别被名字骗了，Sentry 不止看 Crash，PostHog 不止看 PV。

## 核心架构：双核分工

把前端产生的数据分为两类：**"让开发者修 Bug 的"** 和 **"让产品/运营做决策的"**。

| 需求维度 | 核心功能 | 推荐开源组件 |
| --- | --- | --- |
| **APM (稳定性)** | Crash, ANR, 白屏 | **Sentry** |
| **Metric (性能)** | 冷启动耗时, FPS, 页面加载耗时 | **Sentry** (性能版块) |
| **Trace (链路)** | 前端请求 -> 后端 TraceID 串联 | **Sentry** (Network Tracing) |
| **Logs (前端日志)** | 用户行为路径, Console.log, 面包屑 | **Sentry** (Breadcrumbs) + **PostHog** (Session Replay) |
| **Data Analysis** | PV, DAU, 漏斗, 留存 | **PostHog** |
| **A/B Test** | 灰度发布, 功能开关 (Feature Flags) | **PostHog** |

### 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         前端应用 (Flutter/Web)                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
┌────────────────────────────┐   ┌────────────────────────────┐
│         Sentry             │   │         PostHog            │
│  ┌──────────────────────┐  │   │  ┌──────────────────────┐  │
│  │ APM (Crash/ANR)      │  │   │  │ Analytics (PV/DAU)   │  │
│  │ Performance Metrics  │  │   │  │ Session Replay       │  │
│  │ Distributed Tracing  │  │   │  │ Feature Flags        │  │
│  │ Breadcrumbs (Logs)   │  │   │  │ A/B Testing          │  │
│  └──────────────────────┘  │   │  └──────────────────────┘  │
└────────────────────────────┘   └────────────────────────────┘
         │                                    │
         │ 开发者视角                          │ 产品/运营视角
         │ "为什么崩了"                        │ "用户在干什么"
         ▼                                    ▼
┌────────────────────────────┐   ┌────────────────────────────┐
│  稳定性大盘                  │   │  运营大盘                   │
│  - Crash Free Rate         │   │  - DAU/MAU                 │
│  - P99 Latency             │   │  - 转化漏斗                  │
│  - Issues List             │   │  - 功能开关状态              │
└────────────────────────────┘   └────────────────────────────┘
```

## 第一板块：APM + Metric + Trace + Logs (Sentry)

### Logs (日志) & APM

在现代前端监控中，**日志不是 text 文件，而是"面包屑 (Breadcrumbs)"和"上下文"**。

**痛点**：前端不能像后端那样实时把 `print()` 发送到服务器（费流量、费电）。

**Sentry 方案**：
- **自动捕获**：Sentry 会自动 Hook 你的 HTTP 请求、点击事件、路由跳转
- **手动打 Log**：使用 `Sentry.addBreadcrumb()`
- **效果**：当 App 崩溃或报错时，你会看到一张"案发清单"：用户先点了A，再请求了B，最后打印了日志C，然后崩了。这就是前端日志的最佳形态

### Trace (链路追踪)

- **原理**：前端发请求时，自动在 Header 塞入 `sentry-trace` (兼容 W3C Trace Context)
- **Go-Zero 对接**：Go-Zero 的 OTel 中间件会自动识别这个 Header
- **效果**：在 Sentry 后台看"Transaction"，能看到一段瀑布流：`App 点击 (Start)` -> `App HTTP Req` -> `Go-Zero API` -> `DB Query`

### Metric (性能指标)

Sentry 自动采集：
- **Mobile**: 冷/热启动时间、慢帧 (Slow Frames)、冻帧 (Frozen Frames)
- **Web**: LCP, FCP, FID (谷歌核心指标)

### Flutter 代码接入 (Sentry)

```dart
import 'package:sentry_flutter/sentry_flutter.dart';

Future<void> main() async {
  await SentryFlutter.init(
    (options) {
      options.dsn = 'YOUR_SELF_HOSTED_SENTRY_DSN';
      options.tracesSampleRate = 1.0; // 采集 Trace
      options.enableAutoPerformanceTracing = true; // 开启性能监控
      options.attachScreenshot = true; // 崩溃时自动截图
    },
    appRunner: () => runApp(MyApp()),
  );
}

// 业务代码中打日志 (替代 print)
void onUserLogin() {
  // 这条日志会暂存在内存里，如果有报错发生，会随报错一起上传
  Sentry.addBreadcrumb(Breadcrumb(message: '用户点击登录', category: 'auth'));

  // 也可以主动发送普通消息
  Sentry.captureMessage('这是一条普通日志');
}
```

## 第二板块：Data Analysis + A/B Test (PostHog)

前端的数据分析和 A/B 测试，**PostHog** 是目前开源界唯一的全功能选择。

### Data Analysis (数据分析)

- **PV/DAU**: 自动采集
- **自定义事件**: `posthog.capture('buy_success', properties: {'amount': 99})`
- **Session Replay (视觉日志)**: **这是杀手锏**。它会把用户的操作录成视频。如果你想看"日志"，不如直接看用户当时的操作回放，比看 `console.log` 直观一万倍

### A/B Test (A/B 测试与灰度)

- **原理**：在 PostHog 后台配置 Feature Flag (如 `new_home_page`)
- **分流**：可以配置"50% 用户看新版"或"只有 ID 为 10086 的用户看新版"

### Flutter 代码接入 (PostHog)

```dart
import 'package:posthog_flutter/posthog_flutter.dart';

// 1. 埋点 (数据分析)
void trackOrder() {
  Posthog().capture(
    eventName: 'order_placed',
    properties: {'amount': 99.9, 'currency': 'CNY'},
  );
}

// 2. A/B Test (功能开关)
Future<Widget> buildHomePage() async {
  // 从 PostHog 服务器拉取配置，是否开启新 UI
  final bool isNewUI = await Posthog().isFeatureEnabled('new_home_page_v2');

  if (isNewUI) {
    return NewHomePage();
  } else {
    return OldHomePage();
  }
}
```

## 前端可观测性大盘

部署完这套 **Sentry + PostHog** 方案后，你的前端监控平台将包含以下视图：

### A. 稳定性大盘 (Sentry)

- **Crash Free Rate**: 99.9% （最重要的指标）
- **Issues 列表**: 昨晚新增了 5 个崩溃，点击进去能看到堆栈和 Log
- **Performance**: App 平均冷启动耗时 1.2s，其中 Android 低端机拖了后腿

### B. 运营大盘 (PostHog)

- **Traffic**: 实时在线人数 (UV)，今日 DAU
- **Funnels**: 注册转化漏斗（浏览 -> 注册 -> 登录）
- **Feature Flags**: 当前"视频流"功能只对 10% 的用户开启

### C. "前端日志"查看方式

| 场景 | 去哪里看 | 看什么 |
| --- | --- | --- |
| **查 Bug** | **Sentry** | 点开 Issue，看 Breadcrumbs（操作日志）和 Stacktrace（代码行号） |
| **查用户行为** | **PostHog** | 搜索 User ID，点击 **Session Replay**，看他刚才在 App 里到底点了啥导致无法支付 |

## 私有化部署

### Sentry 自托管

```bash
# 使用官方脚本部署
git clone https://github.com/getsentry/self-hosted.git
cd self-hosted
./install.sh
```

### PostHog 自托管

```bash
# 使用 Docker Compose
git clone https://github.com/PostHog/posthog.git
cd posthog
docker compose -f docker-compose.hobby.yml up -d
```

## 总结

```
┌─────────────────────────────────────────────────────────────┐
│                    前端可观测性双核架构                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │       Sentry        │    │      PostHog        │        │
│  │                     │    │                     │        │
│  │  - APM              │    │  - PV / DAU         │        │
│  │  - Trace            │    │  - A/B Test         │        │
│  │  - Metric           │    │  - Feature Flags    │        │
│  │  - Error Logs       │    │  - Session Replay   │        │
│  │                     │    │                     │        │
│  │  开发者视角          │    │  产品/运营视角       │        │
│  └─────────────────────┘    └─────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

这一套组合拳，就是目前开源界前端可观测性的**天花板**，完全满足私有化、灵活性和全功能的需求。
