---
title: Kubernetes (K8s) 核心概念与控制台指南
sidebar_position: 1
tags: [kubernetes, k8s, devops, container]
---

import k8s_dashboard from '@site/static/img/k8s.png'

# Kubernetes (K8s) 核心概念与控制台指南

## 一、 什么是 K8s？

如果说 Docker 是把一个应用程序打包成"集装箱"，那么 Kubernetes (K8s) 就是**负责管理这些集装箱的港口系统**。

它不再关注单台服务器，而是把一堆服务器连起来看作**一台超级计算机**。你不需要告诉它"怎么做"（比如：先启动A，再启动B），你只需要告诉它"我想要什么"（比如：我要 3 个运行中的 Nginx），K8s 会自动帮你达成并维持这个状态。

---

## 二、 核心架构：餐厅隐喻

为了理解 K8s 最重要的三个概念，我们使用"餐厅"作为比喻：

| K8s 概念 | 餐厅比喻 | 职责与特点 |
| --- | --- | --- |
| **Pod** | **服务员/厨师** | **(原子单位)** 真正干活的实体。不仅包含你的代码（Docker容器），还穿了一层 K8s 的马甲（共享网络/存储）。**特点**：不稳定，随时可能生病（崩溃）、离职（被销毁），IP 地址会变。 |
| **Deployment** | **经理/HR** | **(管理策略)** 负责招人和保活。你只跟经理对接，说"我要 3 个服务员"。经理会确保任何时候都有 3 个 Pod 在跑。如果有 Pod 挂了，经理会立马补新的。 |
| **Service** | **前台/总机** | **(流量入口)** 负责对外接待。因为服务员（Pod）一直在换人，位置不固定。Service 提供一个**永远不变的 IP/端口**，把顾客（流量）转发给当前活着的 Pod。 |

**一句话总结关系**：

> **Deployment** 负责生产和管理 **Pod**（干活的），**Service** 负责给这些 Pod 提供一个固定的**入口**（访问的）。

---

## 三、 Dashboard 侧边栏菜单详解

根据你提供的截图，Dashboard 左侧菜单按功能分为了几大类。以下是各选项的详细用途（**加粗**代表最常用）：

### 1. Workloads (工作负载) —— 怎么运行程序

这里定义了程序运行的各种"姿势"。

* **Pods**: 查看当前正在运行的容器实例。排查报错（CrashLoopBackOff）主要看这里。
* **Deployments**: **最常用的方式**。用于部署无状态应用（Web 服务、API）。支持滚动更新、回滚、扩缩容。
* **Stateful Sets**: **部署数据库用**。
* *区别*：Deployment 的 Pod 是像克隆人一样可随便替换的；StatefulSet 的 Pod 有固定编号（如 mysql-0, mysql-1），适合需要持久化存储的应用。


* **Daemon Sets**: **每台机器装一个**。用于部署监控插件、日志收集器等基础设施。
* **Jobs / Cron Jobs**: **任务类**。
* `Jobs`: 跑完就停（如数据备份、一次性计算）。
* `Cron Jobs`: 定时任务（如每天凌晨 3 点清理缓存）。



### 2. Services (服务) —— 怎么访问程序

这里定义了网络流量如何流转。

* **Services**: **内网/简单外网入口**。定义 ClusterIP（集群内访问）或 NodePort（通过节点端口访问）。
* **Ingresses**: **高级网关/域名路由**。相当于 K8s 版的 Nginx 反向代理。它可以把 `api.example.com` 转发给服务 A，把 `www.example.com` 转发给服务 B。

### 3. Config and Storage (配置与存储) —— 数据去哪

这里实现了代码与配置/数据的解耦。

* **Config Maps**: **存明文配置**。比如 `nginx.conf` 或环境变量。
* **Secrets**: **存敏感信息**。比如数据库密码、SSL 证书。内容会被加密或编码。
* **Persistent Volume Claims (PVC)**: **申请硬盘**。因为 Pod 重启后文件会消失，如果需要保存数据（如 MySQL 数据文件），必须申请 PVC 挂载到 Pod 上。

### 4. Cluster (集群管理) —— 基础设施

* **Namespaces**: 资源隔离的"房间"（详见下文）。
* **Nodes**: 查看物理服务器/虚拟机的健康状态（CPU/内存占用）。
* **Roles / Role Bindings**: 权限控制（RBAC），定义谁能操作什么资源。

---

## 四、 Namespaces (命名空间) 详解

Namespace 是 K8s 用来隔离资源的"虚拟文件夹"。你在截图中看到的几个默认 Namespace 含义如下：

1. **`default`**: **(主工作区)** 如果你不指定 Namespace，你部署的所有应用都会默认进到这里。
2. **`kube-system`**: **(系统核心)** K8s 自身的组件（DNS、网络插件、监控）都在这里。**绝对不要随意删除这里的东西**。
3. **`kubernetes-dashboard`**: **(工具区)** 你当前看到的这个 Dashboard 网页服务本身就运行在这个空间里。
4. **`kube-public`**: **(公共区)** 存放集群公共信息（如根证书），通常对所有用户可见。
5. **`kube-node-lease`**: **(心跳区)** 用于存放节点向控制面发送心跳记录的对象，数量虽多但体积极小，用于判断节点是否存活。

---

## 五、 操作核心：`kubectl apply -f`

K8s 的操作哲学是 **"声明式 API"**。

* **传统做法 (命令式)**：一步步下指令。
* `run container` -> `connect network` -> `start app`


* **K8s 做法 (声明式)**：提交一张"图纸"（YAML 文件）。
* 命令：`kubectl apply -f my-app.yaml`
* 逻辑：你只需要在 YAML 里写清楚"我要 3 个副本，用 v1.0 镜像，开放 80 端口"。K8s 会自动对比**现状**和**期望**。
* 如果是第一次，它会创建。
* 如果你改了 YAML（比如 3 改成 5），它会补齐差额。
* 如果现状和 YAML 一样，它什么都不做。





**总结**：`kubectl apply -f` 是 K8s 的万能钥匙，建议所有正式变更都通过修改 YAML 文件并重新 Apply 来完成，而不是用临时命令。

---

## Dashboard 界面说明

下图展示了 Kubernetes Dashboard 的主要界面，包括工作负载、Pod 状态、资源使用情况等信息：

<img src={k8s_dashboard} alt="Kubernetes Dashboard 界面" />

通过 Dashboard，你可以直观地查看和管理集群中的各种资源，包括：
- 工作负载的运行状态
- Pod 的 CPU 和内存使用情况
- 服务的网络配置
- 配置和存储资源

---

## 总结

Kubernetes 是一个强大的容器编排系统，通过声明式的方式管理应用的生命周期。掌握 Pod、Deployment、Service 这三个核心概念，以及 Dashboard 的使用方法，是入门 K8s 的关键。记住：**Deployment 管理 Pod，Service 提供访问入口**，这是理解 K8s 的基础。

