---
title: Jenkins 基础架构与容器化部署
sidebar_position: 1
tags: [jenkins, docker, devops, ci]
---

# Jenkins 基础架构与容器化部署

## 1. 核心概念

Jenkins 是基于 Java 的开源自动化服务器，核心作用是 **CI (持续集成)**。它通过"插件"发出指令，通过"节点/代理"执行命令。

## 2. 部署架构：Docker out of Docker (DooD)

**误区纠正：** Jenkins 容器内部**不包含** Docker 引擎，也不能运行子容器。

**正确做法：** 采用"兄弟容器"模式。Jenkins 容器通过挂载宿主机的 `docker.sock`，指挥宿主机启动兄弟容器来执行构建任务。

```
┌─────────────────────────────────────────────────────┐
│                    宿主机                            │
│  ┌─────────────────┐    ┌─────────────────┐         │
│  │ Jenkins 容器    │    │ 构建容器 (兄弟)  │         │
│  │ (Docker CLI)   │───▶│ (Maven/Node等)  │         │
│  └────────┬────────┘    └─────────────────┘         │
│           │                                         │
│           ▼                                         │
│  ┌─────────────────┐                                │
│  │ docker.sock     │ ◀── Docker 引擎                │
│  └─────────────────┘                                │
└─────────────────────────────────────────────────────┘
```

## 3. Docker 启动命令 (生产级)

### 3.1 自定义 Dockerfile

首先需要构建一个包含 Docker CLI 和 Kubectl 的自定义镜像：

```dockerfile
FROM jenkins/jenkins:lts

USER root

# 安装 Docker CLI
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli

# 安装 Kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl

# 配置 Docker 组权限
RUN groupadd -g 999 docker && usermod -aG docker jenkins

USER jenkins
```

### 3.2 启动命令

```bash
docker run -d \
  -u root \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins-data:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --name jenkins \
  my-custom-jenkins-image
```

**核心参数解析：**

| 参数 | 说明 |
|------|------|
| `-v /var/run/docker.sock:/var/run/docker.sock` | 赋予 Jenkins 指挥宿主机 Docker 的能力 |
| `-v jenkins-data:/var/jenkins_home` | 持久化 Jenkins 配置和任务数据 |
| `-p 8080:8080` | Web UI 端口 |
| `-p 50000:50000` | Agent 通信端口 |

## 4. 必备插件清单

不要贪多，安装以下核心插件即可：

| 插件名称 | 功能说明 |
|----------|----------|
| **Pipeline** | 核心流水线功能 |
| **Docker Pipeline** | 允许在 Pipeline 中使用 `agent { docker ... }` |
| **Kubernetes** | 允许动态创建 K8s Pod 作为从节点 |
| **AnsiColor** | 让构建日志显示颜色 |
| **Blue Ocean** | 提供现代化的可视化 UI |
| **Rebuilder** | 方便带参数的任务重新构建 |
| **Git** | Git 源码管理支持 |
| **Credentials** | 凭据管理 |

## 5. 初始化配置

### 5.1 获取初始密码

```bash
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

### 5.2 配置建议

1. **安装推荐插件** - 首次启动时选择 "Install suggested plugins"
2. **创建管理员账户** - 不要使用 admin/admin
3. **配置 URL** - 设置 Jenkins URL 为实际访问地址
4. **配置安全** - 启用 CSRF 保护，配置授权策略

## 6. 常见问题排查

### 6.1 权限问题

如果遇到 Docker 权限问题：

```bash
# 检查 docker.sock 权限
ls -la /var/run/docker.sock

# 容器内检查 jenkins 用户组
docker exec jenkins id jenkins
```

### 6.2 磁盘空间

Jenkins 会积累大量构建历史，定期清理：

```bash
# 清理旧的构建记录
# 在 Jenkins 管理界面设置构建保留策略

# 清理 Docker 缓存
docker system prune -af
```
