---
title: kubectl 与 Helm 入门指南
sidebar_position: 1
tags: [kubernetes, kubectl, helm, devops]
---

# Kubernetes 管理双雄：kubectl 与 Helm 入门指南

在 Kubernetes 的世界里，我们需要通过工具与集群进行交互。最基础、最核心的工具是 **kubectl**，而为了解决应用交付和管理的复杂性，**Helm** 应运而生。

简而言之：

- **kubectl** 是手工打造（操作单个砖块）。
- **Helm** 是预制件组装（管理整套蓝图）。

---

## 1. kubectl：Kubernetes 的"瑞士军刀"

### 什么是 kubectl？

`kubectl` 是 Kubernetes 的官方命令行工具（CLI）。它直接与 Kubernetes 的 **API Server** 通信，用于发送指令、查询状态和管理集群资源。

它是你与 K8s 集群对话的"原生语言"。无论你是要查看日志、创建一个 Pod，还是调试网络问题，kubectl 都是必不可少的。

### 核心功能

- **集群管理**：查看节点状态、组件健康状况。
- **资源操作**：创建、更新、删除 K8s 资源（Pods, Services, Deployments 等）。
- **调试与诊断**：查看容器日志、进入容器内部、查看资源详细描述。

### 常用命令示例

| 操作 | 命令 | 说明 |
| --- | --- | --- |
| **查看资源** | `kubectl get pods` | 列出当前命名空间下的所有 Pod |
| **查看详情** | `kubectl describe pod <pod-name>` | 查看某个 Pod 的详细事件和配置（调试神器） |
| **查看日志** | `kubectl logs <pod-name>` | 查看容器输出的日志 |
| **应用配置** | `kubectl apply -f deployment.yaml` | 根据 YAML 文件创建或更新资源（声明式） |
| **进入容器** | `kubectl exec -it <pod-name> -- bash` | 就像 SSH 一样进入容器内部 |

---

## 2. Helm：Kubernetes 的"包管理器"

### 什么是 Helm？

如果说 Kubernetes 是操作系统，那么 **Helm** 就是它的包管理器（类似于 Linux 上的 `apt`、`yum` 或 macOS 上的 `brew`，Node.js 的 `npm`）。

在 K8s 中，部署一个复杂的应用（如一个数据库集群）可能涉及数十个 YAML 文件（Deployment, Service, ConfigMap, Secret, PVC 等）。Helm 将这些文件打包成一个整体，称为 **Chart**，从而实现一键部署和管理。

### 核心概念

1. **Chart（图表）**：Helm 的包。它包含了一组定义 K8s 资源的模板文件。
2. **Repository（仓库）**：存放 Chart 的地方，类似于软件源。
3. **Release（发行版）**：Chart 在 K8s 集群中运行的一个实例。同一个 Chart 可以多次安装，每次都会生成一个新的 Release。
4. **Values（值）**：配置文件（通常是 `values.yaml`），允许你在安装时自定义配置（如修改副本数、镜像版本、端口号），而无需修改 Chart 本身的代码。

### 常用命令示例

| 操作 | 命令 | 说明 |
| --- | --- | --- |
| **添加仓库** | `helm repo add bitnami https://...` | 添加第三方 Chart 仓库 |
| **搜索应用** | `helm search repo nginx` | 在仓库中查找 Nginx 包 |
| **安装应用** | `helm install my-nginx bitnami/nginx` | 将 Nginx 安装到集群，命名为 "my-nginx" |
| **自定义安装** | `helm install -f values.yaml ...` | 使用自定义配置文件安装 |
| **列出应用** | `helm list` | 查看当前集群安装了哪些 Helm 应用 |
| **卸载应用** | `helm uninstall my-nginx` | 一键删除该应用的所有关联资源 |

---

## 3. kubectl vs Helm：有什么区别？

虽然它们都操作 K8s 资源，但关注的层级不同。

| 特性 | kubectl | Helm |
| --- | --- | --- |
| **定位** | 底层操作工具 | 高级包管理工具 |
| **管理粒度** | **微观**：关注单个资源（Pod, Service） | **宏观**：关注整个应用（App Stack） |
| **文件管理** | 直接管理静态的 YAML 文件 | 管理包含动态模板的 Chart 包 |
| **复用性** | 较低（需要手动修改 YAML） | 极高（通过 `values.yaml` 注入参数） |
| **版本控制** | 需配合 Git 使用，回滚较繁琐 | 内置版本管理，`helm rollback` 可一键回滚 |
| **类比** | 手工砌砖盖房 | 购买预制房屋套件 |

### 什么时候用谁？

**使用 kubectl 当...**

- 你需要快速查看集群状态或某个 Pod 为什么挂了。
- 你需要临时修改某个配置或删除某个特定资源。
- 你正在学习 K8s 的基础原理。
- 你的应用非常简单（例如只有一个 YAML 文件）。

**使用 Helm 当...**

- 你需要部署复杂的应用（包含多个依赖服务）。
- 你需要将应用分享给其他团队，让他们能通过简单的配置一键安装。
- 你需要管理应用的生命周期（版本升级、一键回滚）。
- 你需要为不同的环境（开发、测试、生产）部署相同的应用，但配置参数不同。

---

## 4. 实战对比：部署 Nginx 应用

假设你的任务是：部署一个 **Nginx** 应用，使用镜像版本 `1.19.0`，并且需要启动 `3` 个副本。

### 方式一：使用 kubectl（静态文件流）

这种方式就像是在写死板的"硬代码"。

**1. 编写 `deployment.yaml` 文件：**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-server
spec:
  replicas: 3               # <--- 这里的 3 是写死的
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.19.0 # <--- 这里的版本号是写死的
        ports:
        - containerPort: 80
```

**2. 部署命令：**

```bash
kubectl apply -f deployment.yaml
```

**3. 痛点场景：**

如果第二天，老板让你把镜像换成 `1.20.0`，并且副本数改成 `5` 个：

- 你必须**打开这个文件**。
- 找到第 6 行，把 `3` 改成 `5`。
- 找到第 17 行，把 `1.19.0` 改成 `1.20.0`。
- 保存，再次运行 `kubectl apply`。

### 方式二：使用 Helm（动态模板流）

这种方式就像是在做"填空题"。我们把写死的地方挖成坑（变量）。

**1. 编写模板文件 `templates/deployment.yaml`：**

注意看双大括号 `{{ }}` 的部分，这是 Helm 的语法。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-server
spec:
  replicas: {{ .Values.replicaCount }}  # <--- 这是一个变量坑位
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: "{{ .Values.image.repo }}:{{ .Values.image.tag }}" # <--- 这也是变量坑位
        ports:
        - containerPort: 80
```

**2. 准备默认配置文件 `values.yaml`：**

```yaml
# 这里定义默认填什么
replicaCount: 3
image:
  repo: nginx
  tag: 1.19.0
```

**3. 部署命令：**

- **场景 A：正常部署（使用默认值）**

```bash
# Helm 会自动读取 values.yaml 填入模板
helm install my-web ./my-chart-folder
```

- **场景 B：老板需求变更（改版本和副本数）**

你**完全不需要**去修改任何 YAML 文件，直接在命令行里通过 `--set` 覆盖变量：

```bash
helm upgrade my-web ./my-chart-folder --set image.tag=1.20.0 --set replicaCount=5
```

---

## 5. 为什么 Helm 对 CI/CD 极其重要？

在 Jenkins 或 GitLab CI 流水线中，机器是很难去"打开文件修改第几行"的。

使用 Helm，你的流水线脚本可以是通用的：

```bash
# 比如流水线里有个变量叫 $BUILD_VERSION
helm upgrade my-app ./charts --set image.tag=$BUILD_VERSION
```

这样，无论你用 kubectl 还是 Helm，最终跑在集群里的都是你打的那个 Docker 镜像。它们的区别在于 **"如何告诉 Kubernetes 去拉取并运行这个镜像"**。

---

## 6. 总结

**kubectl** 和 **Helm** 并不是互斥的，而是互补的。

在现代 DevOps 流程中，通常的模式是：

1. 开发人员编写代码和 **Helm Chart**。
2. CI/CD 流水线使用 **Helm** 将应用部署到集群。
3. 运维/SRE 人员使用 **kubectl** 进行日常监控、故障排查和底层维护。

> **一句话总结**：用 Helm 来"买"和"装"家具，用 kubectl 来"修"家具和"打扫"房间。

| 环节 | kubectl 部署 | Helm 部署 |
| --- | --- | --- |
| **Docker 镜像** | **一样**（都需要先打好镜像） | **一样**（都需要先打好镜像） |
| **部署文件** | 静态的 YAML (写死 `image: my-app:v1`) | 动态的模板 (写成 `image: {{ .Values.tag }}`) |
| **更新版本** | 手动修改 YAML 文件里的 tag，再 apply | 命令行直接传参 `--set image.tag=v2` |
| **多环境** | 需要复制多份 YAML (dev.yaml, prod.yaml) | 同一套 Chart，通过不同的 `values.yaml` 区分 |

**结论：**

- **kubectl**：你手动修改文件，告诉 K8s "我要这个具体的配置"。
- **Helm**：你写好逻辑，通过命令行传入参数，Helm 帮你生成具体的配置并告诉 K8s。
