---
title: 可观测性指标体系设计
sidebar_position: 6
tags: [observability, metrics, sre, red, use, grafana, alerting, prometheus]
---

# 生产级可观测性指标体系设计

构建一套生产级可观测性体系，核心不在于"指标越多越好"，而在于**"分层监控，重点突出"**。

根据 **Google SRE** 推崇的 **"RED"** 和 **"USE"** 方法论，准备以下 **4 个层级** 的核心指标。这套指标体系配合 Go-Zero + OpenTelemetry + K8s 简直是绝配。

## 第一层：业务关键指标 (Business Metrics)

**这是给老板和产品经理看的。** 如果这层挂了，技术指标再好看都没用。

| 指标名称 | 含义 | 为什么重要 | 数据来源 |
| --- | --- | --- | --- |
| **核心业务量** | 如：每分钟下单数、支付成功数、短信发送数 | 直观反映业务是否"活着"。如果突然归零，这就是最高级(P0)事故 | 手动埋点 (OTel Metrics) |
| **业务成功率** | (成功订单 / 总请求) | 排除 HTTP 500，业务逻辑上的失败（如余额不足、库存不足）也要关注 | 手动埋点 |
| **业务转换漏斗** | 浏览 -> 加购 -> 下单 -> 支付 | 监控每一步的流失率是否异常 | 前端/后端埋点 (PostHog/Umami) |

## 第二层：服务运行指标 (Application Metrics)

**这是给后端开发看的。** 这里遵循 **RED 方法论** (Rate, Errors, Duration)。

:::tip
Go-Zero 框架会自动输出这些指标，不用你写代码。
:::

| 指标维度 | 具体指标 | 告警阈值建议 (参考) |
| --- | --- | --- |
| **R (Rate)** | **QPS (Requests Per Second)** | 环比突增/突降 50% 告警（防攻击或上游故障） |
| **E (Errors)** | **HTTP 5xx 比例** | > 1% 告警 |
|  | **RPC 错误率** | > 0.5% 告警（微服务内部调用失败） |
| **D (Duration)** | **P99 延迟** (99%的请求有多快) | > 500ms (根据业务调整)。只看平均值没意义，要看长尾 |
|  | **P95 延迟** | > 200ms |
| **Runtime** | **Goroutine 数量** | 如果持续上涨不降，说明有协程泄漏 (Leak) |
|  | **GC Pauses (垃圾回收暂停)** | 如果 GC > 100ms，说明对象分配太频繁，服务会卡顿 |

### RED 方法论图解

```
┌─────────────────────────────────────────────────────────────┐
│                    RED 方法论                                │
├─────────────────────────────────────────────────────────────┤
│  R (Rate)      │  请求速率 - 每秒多少请求                     │
│  E (Errors)    │  错误数量 - 每秒多少请求失败                  │
│  D (Duration)  │  耗时分布 - 请求响应时间分布                  │
└─────────────────────────────────────────────────────────────┘
```

## 第三层：中间件/依赖指标 (Middleware Metrics)

**这是由运维/架构师重点关注的。** 很多时候服务慢，是因为数据库或 Redis 慢。

### 1. 数据库 (MySQL/Postgres)

| 指标 | 说明 |
| --- | --- |
| **连接池使用率** | 如果连接池（Connection Pool）满了，应用会直接报错 |
| **慢查询数量 (Slow Queries)** | 每分钟产生的慢 SQL 条数 |
| **主从延迟 (Replication Lag)** | 写主库，读从库，如果延迟大，用户会发现刚改完的数据没变 |

### 2. 缓存 (Redis)

| 指标 | 说明 |
| --- | --- |
| **缓存命中率 (Hit Rate)** | 如果低于 80%，说明缓存穿透了，压力全给到了 DB |
| **内存使用率** | Redis 满了会根据策略踢人（Evict），导致数据丢失或报错 |
| **网络带宽** | Redis 吞吐量极高，容易把网卡打满 |

### 3. 消息队列 (Kafka/RocketMQ)

| 指标 | 说明 |
| --- | --- |
| **Consumer Lag (消费积压)** | **这是最重要的指标**。生产太快，消费太慢，导致消息堆积。如果 Lag 持续增加，说明消费者处理不过来了 |

## 第四层：基础设施指标 (Infrastructure Metrics - K8s)

**这是底座。** 遵循 **USE 方法论** (Utilization, Saturation, Errors)。

### USE 方法论图解

```
┌─────────────────────────────────────────────────────────────┐
│                    USE 方法论                                │
├─────────────────────────────────────────────────────────────┤
│  U (Utilization)  │  资源利用率 - CPU/Memory 使用百分比       │
│  S (Saturation)   │  饱和度 - 资源排队/等待情况               │
│  E (Errors)       │  错误数 - 资源操作失败次数                │
└─────────────────────────────────────────────────────────────┘
```

### K8s 核心指标

| 指标名称 | 含义 | 致命性 |
| --- | --- | --- |
| **Pod OOM Killed** | **内存溢出被杀** | ⭐⭐⭐⭐⭐ (Pod 会无限重启，服务极不稳定) |
| **CPU Throttling** | **CPU 被限流** | ⭐⭐⭐ (K8s 限制了你的 CPU 使用，导致服务变慢) |
| **Node Disk Usage** | 节点磁盘空间 | ⭐⭐⭐⭐ (磁盘满会导致 Docker 无法写日志，Pod 驱逐) |
| **Network I/O** | 进出流量 | ⭐⭐ (除了视频/下载业务，一般不会把千兆网卡打满) |

## 落地实战：Grafana 大盘设计

不要把几百个指标堆在一起。建议建立 **3 个层级的大盘 (Dashboard)**：

### 1. 全局概览大盘 (Global Overview)

- **受众**: 技术总监 / On-call 人员
- **内容**:
  - 全站核心业务 QPS 曲线
  - 核心链路（如：下单接口）的成功率 & P99 延迟
  - 当前告警数量（红绿灯显示）
- **一句话**: *"一眼看出系统有没有崩。"*

### 2. 服务详情大盘 (Service Detail)

- **受众**: 具体的微服务负责人（比如订单组开发）
- **内容**: (Go-Zero 自动生成的那些)
  - 该服务的 QPS, Error Rate, P99 Latency
  - 该服务的 Goroutine 数量, Heap 内存
  - 该服务调用的 DB 和 Redis 的耗时

### 3. 资源排查大盘 (Pod/Node Level)

- **受众**: 运维 / SRE
- **内容**:
  - 具体某个 Pod 的 CPU/Memory 曲线
  - K8s 节点的负载情况

## 必须配置告警的 P0 级指标

如果你没时间配那么多，**先把下面这 5 个配上告警**，今晚就能睡个安稳觉：

:::danger 关键告警清单
1. **HTTP/RPC 错误率 > 5%** (服务挂了)
2. **P99 延迟 > 2秒** (服务卡死了)
3. **Pod OOM Killed (Count > 0)** (内存爆了)
4. **Kafka Consumer Lag > 10000** (消息处理不过来了)
5. **核心业务量(如订单) 跌零** (即使系统没报错，业务也可能断了)
:::

## 告警配置示例 (Prometheus AlertManager)

```yaml
groups:
  - name: critical-alerts
    rules:
      # 错误率告警
      - alert: HighErrorRate
        expr: |
          sum(rate(http_server_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_server_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "HTTP 错误率超过 5%"
          description: "当前错误率: {{ $value | humanizePercentage }}"

      # P99 延迟告警
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_server_request_duration_seconds_bucket[5m])) by (le)
          ) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 延迟超过 2 秒"

      # OOM 告警
      - alert: PodOOMKilled
        expr: |
          kube_pod_container_status_last_terminated_reason{reason="OOMKilled"} > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Pod {{ $labels.pod }} OOM Killed"

      # Kafka 消费积压
      - alert: KafkaConsumerLag
        expr: kafka_consumer_lag > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Kafka 消费积压: {{ $value }}"
```

## 总结：分层监控架构

```
┌─────────────────────────────────────────────────────────────┐
│                    第一层: 业务指标                          │
│            (核心业务量、成功率、转换漏斗)                      │
├─────────────────────────────────────────────────────────────┤
│                    第二层: 服务指标 (RED)                    │
│              (QPS、错误率、P99/P95延迟)                      │
├─────────────────────────────────────────────────────────────┤
│                    第三层: 中间件指标                         │
│           (DB连接池、Redis命中率、Kafka Lag)                 │
├─────────────────────────────────────────────────────────────┤
│                    第四层: 基础设施指标 (USE)                 │
│              (OOM、CPU Throttling、磁盘)                     │
└─────────────────────────────────────────────────────────────┘
```

这套指标体系的核心思想是：**从业务到基础设施，自上而下分层，每层都有明确的关注点和责任人**。
