---
title: Kubernetes 热更新与自动扩缩容 (HPA)
sidebar_position: 3
tags: [kubernetes, k8s, rolling-update, hpa, autoscaling, devops]
---

# Kubernetes 核心机制详解：热更新与自动扩缩容

这两项功能是 K8s 区别于传统运维模式的核心竞争力，也是保障服务"高可用"和"高弹性"的基石。

## 1. 热更新 (Rolling Update)

### 1.1 什么是热更新？

热更新（Rolling Update）是 Kubernetes `Deployment` 默认的发布策略。它的目标是在**不中断服务**的情况下，将应用从旧版本（v1）平滑过渡到新版本（v2）。

**核心理念**：新老交替，温水煮青蛙。K8s 不会一次性杀掉所有旧容器，而是**启动一个新容器 -> 确认存活 -> 杀掉一个旧容器**，如此循环，直到全部替换完成。

### 1.2 核心原理：ReplicaSet 的交接棒

当我们更新 Deployment 的镜像时，K8s 底层实际发生了以下动作：

1. Deployment 创建一个新的 **ReplicaSet (v2)**，初始副本数为 0。
2. Deployment 指挥 v2 ReplicaSet 增加副本（扩容）。
3. 当 v2 的 Pod 准备就绪（Ready）后，Deployment 指挥旧的 **ReplicaSet (v1)** 减少副本（缩容）。
4. 重复上述步骤，直到 v1 副本数为 0，v2 副本数达到期望值。

### 1.3 关键配置参数

在 Deployment 的 YAML 中，通过 `strategy` 字段控制更新节奏：

```yaml
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # 激增数：允许比期望副本数多几个？
      maxUnavailable: 0  # 不可用数：允许发布过程中少几个？
```

- **maxSurge (最大激增数)**：
  - 设置为 `2` 或 `20%`。表示在发布过程中，最多可以存在 `10 + 2 = 12` 个 Pod。
  - *作用*：先启动新 Pod，确保资源足够后再删旧的。

- **maxUnavailable (最大不可用数)**：
  - 设置为 `0`。表示在发布过程中，**任何时刻**可用的 Pod 数量都不能少于 10 个。
  - *作用*：保证服务处理能力不下降，适合对稳定性要求极高的场景。

### 1.4 必不可少的机制：Readiness Probe (就绪探针)

**这是热更新成功的关键！**

如果没配探针，Pod 一启动（容器进程跑起来），K8s 就认为它"好了"，立马切断旧 Pod。如果新程序启动慢（比如 Java 加载 Spring），流量打进来就会报错。

**配置示例**：

```yaml
spec:
  containers:
  - name: my-app
    image: my-app:v2
    # 告诉 K8s：我不只活了，我还准备好接客了
    readinessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 5  # 启动后等5秒再开始查
      periodSeconds: 10       # 每10秒查一次
```

- **逻辑**：只有当 `readinessProbe` 检测成功（返回 200），K8s 才会把这个新 Pod 标记为 `Ready`，才会把流量切给它，并开始删除旧 Pod。

### 1.5 常用命令

```bash
# 触发更新
kubectl set image deployment/myapp nginx=nginx:1.19

# 查看进度
kubectl rollout status deployment/myapp

# 后悔了（回滚）
kubectl rollout undo deployment/myapp
```

---

## 2. 自动扩缩容 (HPA - Horizontal Pod Autoscaler)

### 2.1 什么是 HPA？

HPA 是 Kubernetes 的"弹性伸缩控制器"。它像一个不知疲倦的管理员，每隔 15 秒（默认）检查一次 Pod 的负载情况。

- **负载高**：自动增加 Pod 数量，分摊压力。
- **负载低**：自动减少 Pod 数量，节省服务器资源（省钱）。

### 2.2 工作流程 (The Control Loop)

1. **收集数据**：`Metrics Server` 从各个 Pod 收集 CPU 和内存使用率。
2. **计算需求**：HPA 控制器根据你设定的公式计算所需的 Pod 数量。
3. **执行指令**：HPA 修改 Deployment 的 `replicas` 数量。
4. **最终落地**：Deployment 负责创建或销毁 Pod。

### 2.3 核心算法

HPA 使用以下公式来决定扩容还是缩容：

$$
期望副本数 = \lceil 当前副本数 \times \frac{当前指标值}{目标指标值} \rceil
$$

**举例**：

- 当前有 **2** 个 Pod。
- 你设定目标 CPU 使用率是 **50%**。
- 现在突发流量，每个 Pod 的 CPU 飙升到了 **90%**。
- 计算：$\lceil 2 \times \frac{90\%}{50\%} \rceil = \lceil 3.6 \rceil = 4$
- **结果**：K8s 会将 Pod 扩容到 4 个。

### 2.4 实施 HPA 的硬性前提

要使用 HPA，你必须做两件事：

1. **集群必须安装 Metrics Server**：这是 HPA 的眼睛，没有它 HPA 读不到 CPU 数据。
2. **Deployment 必须配置 Request 资源限制**：HPA 是根据百分比计算的，如果你没告诉 K8s 分母（Request）是多少，它算不出百分比。

**YAML 配置示例**：

**第一步：在 Deployment 里定义"分母"**

```yaml
containers:
- name: php-apache
  image: k8s.gcr.io/hpa-example
  resources:
    requests: # 必须配置这个！
      cpu: 200m  # 200毫核（0.2核）
```

**第二步：创建 HPA 规则**

```yaml
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: php-apache-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: php-apache
  minReplicas: 1   # 无论多闲，至少保留1个
  maxReplicas: 10  # 无论多忙，最多开10个（防止资源耗尽）
  targetCPUUtilizationPercentage: 50 # 维持在 50% 利用率
```

### 2.5 扩缩容的"防抖动"机制

- **扩容（Scale Up）**：非常激进。一旦检测到负载超标，几分钟内就会把 Pod 加满，为了快速抗住流量。
- **缩容（Scale Down）**：非常保守。默认有一个 **5分钟的冷却期 (Stabilization Window)**。
  - *为什么？* 防止流量忽高忽低导致 Pod 刚删掉又要创建（抖动）。只有持续 5 分钟负载都很低，K8s 才会真的开始删 Pod。

---

## 3. 两者如何配合工作？

在生产环境中，**热更新**和**HPA**是同时工作的，可能会出现这种情况：

> *正在发新版本（热更新中），突然流量暴涨（触发 HPA），会打架吗？*

**答案是：不会，它们配合得很好。**

1. **HPA 拥有最高指挥权**：HPA 并不关心你用的是 v1 还是 v2 镜像，它只看 CPU。如果 CPU 高了，它直接修改 Deployment 的 `replicas` 总数（比如从 3 改到 6）。
2. **Deployment 负责执行**：Deployment 收到 `replicas=6` 的指令后，会根据当前的滚动更新策略，按比例去增加 v1 或 v2 的 Pod，最终达到总数 6 个。

---

## 最佳实践总结

| 功能 | 核心作用 | 关键配置项 | 注意事项 |
| --- | --- | --- | --- |
| **热更新** | 平滑发布代码，零停机 | `maxSurge`, `maxUnavailable` | **一定要配 Readiness Probe**，否则等于是在裸奔，容易发布即崩溃。 |
| **HPA** | 根据流量自动伸缩，抗压且省钱 | `minReplicas`, `maxReplicas`, `targetCPU` | **一定要配 resources.requests**，否则 HPA 不生效。 |
