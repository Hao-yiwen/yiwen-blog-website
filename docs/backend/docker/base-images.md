---
title: Docker 常用基础镜像对比
sidebar_position: 1
tags: [docker, container, devops]
---

# Docker 常用基础镜像对比

Docker Hub 上有大量官方和社区维护的镜像。了解这些镜像的特点和用途，能帮助你快速构建容器化应用。

## 1. 操作系统基座 (Base OS)

构建自定义镜像时，通常需要从这些基础镜像开始：

### Alpine

```bash
docker pull alpine
```

| 特点 | 说明 |
|------|------|
| 体积 | 极小（约 5MB） |
| 安全性 | 高 |
| 包管理器 | apk |
| C 库 | musl libc |

**用途：** 生产环境首选，用于构建尽可能轻量级的容器。

### Ubuntu / Debian

```bash
docker pull ubuntu
docker pull debian
```

| 特点 | 说明 |
|------|------|
| 体积 | 较大（约 70-120MB） |
| 包管理器 | apt |
| C 库 | glibc |
| 生态 | 完整的软件仓库 |

**用途：** 开发环境，或需要依赖某些 Alpine 不支持的 glibc 库的场景。

### Busybox

```bash
docker pull busybox
```

| 特点 | 说明 |
|------|------|
| 体积 | 极小（约 1-5MB） |
| 定位 | "嵌入式 Linux 的瑞士军刀" |
| 工具 | 集成常用 Unix 工具 |

**用途：** 调试、网络测试、初始化容器（init containers）。

## 2. Web 服务器与代理 (Web Servers & Proxies)

用于托管静态网站或作为应用的反向代理：

### Nginx

```bash
docker pull nginx
```

**用途：** 静态资源服务器、负载均衡、反向代理。Docker 中最流行的镜像之一。

### Apache HTTP Server

```bash
docker pull httpd
```

**用途：** 老牌 Web 服务器，常用于运行传统的 PHP 应用或静态站点。

### Traefik

```bash
docker pull traefik
```

**用途：** 专为微服务设计的云原生反向代理，能自动发现 Docker 容器并配置路由。

## 3. 数据库 (Databases)

容器化数据库可以极大地简化开发和测试流程：

### MySQL / MariaDB

```bash
docker pull mysql
docker pull mariadb
```

**用途：** 最通用的关系型数据库。

**启动示例：**

```bash
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=my-secret-pw \
  -p 3306:3306 \
  mysql:8.0
```

### PostgreSQL

```bash
docker pull postgres
```

**用途：** 功能强大的开源对象关系数据库，深受开发者喜爱。

### MongoDB

```bash
docker pull mongo
```

**用途：** 最流行的 NoSQL 文档数据库，适合灵活的数据结构。

## 4. 缓存与消息队列 (Caching & Messaging)

用于提升应用性能和解耦服务：

### Redis

```bash
docker pull redis
```

**用途：** 内存数据结构存储，用作数据库、缓存和消息代理。

### RabbitMQ

```bash
docker pull rabbitmq:management
```

**用途：** 广泛使用的消息代理（Message Broker），支持多种消息协议。

:::tip
使用 `rabbitmq:management` 标签可以获得自带的 Web 管理界面（端口 15672）。
:::

## 5. 编程语言运行时 (Language Runtimes)

如果你不想在本地安装繁琐的语言环境，或者需要构建应用镜像：

| 语言 | 镜像名称 | 示例标签 |
|------|----------|----------|
| Python | `python` | `python:3.12-slim` |
| Node.js | `node` | `node:20-alpine` |
| Java | `eclipse-temurin` | `eclipse-temurin:21-jdk` |
| Go | `golang` | `golang:1.22-alpine` |

## 6. 运维与监控工具 (DevOps & Monitoring)

| 工具 | 镜像名称 | 用途 |
|------|----------|------|
| Jenkins | `jenkins/jenkins` | 自动化 CI/CD 服务器 |
| Prometheus | `prom/prometheus` | 系统监控和报警工具 |
| Grafana | `grafana/grafana` | 数据可视化平台 |
| Portainer | `portainer/portainer-ce` | 可视化 Docker 容器管理工具 |

## 镜像标签选择指南

在拉取镜像时（例如 `docker pull python:<tag>`），标签的选择非常关键：

| 标签后缀 | 含义 | 适用场景 |
|----------|------|----------|
| `latest` | 默认标签，通常是最新版本 | ❌ 不推荐在生产环境使用，版本不可控 |
| `slim` | 精简版 (如 `python:3.9-slim`) | ✅ 去除非必要构建工具，适合大多数部署 |
| `alpine` | 基于 Alpine Linux | ✅ 体积最小，但可能有 musl libc 兼容性问题 |
| 具体版本号 | 如 `mysql:8.0.33` | ✅ **强烈推荐**，锁定版本保证稳定性 |

:::warning 关于 Alpine 兼容性
Alpine 使用 musl libc 而非 glibc，某些依赖 C 语言扩展的库可能会有兼容性问题。如果遇到问题，可以切换到 `slim` 变体。
:::

## 镜像大小对比

以 Python 为例，不同变体的体积差异：

| 变体 | 大小（约） |
|------|-----------|
| `python:3.12` | ~1GB |
| `python:3.12-slim` | ~150MB |
| `python:3.12-alpine` | ~50MB |

## 最佳实践

1. **生产环境锁定版本**：使用具体版本号而非 `latest`
2. **优先选择轻量镜像**：`alpine` 或 `slim` 变体
3. **多阶段构建**：减少最终镜像体积
4. **定期更新基础镜像**：获取安全补丁

```dockerfile
# 多阶段构建示例
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o main .

FROM alpine:3.19
COPY --from=builder /app/main /main
CMD ["/main"]
```
