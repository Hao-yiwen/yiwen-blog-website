---
title: 企业级制品管理
sidebar_position: 4
tags: [artifactory, nexus, harbor, devops, registry]
---

# 企业级制品管理 (Artifactory/Nexus)

## 1. 为什么要用 Artifactory？

相比于简陋的 `docker-registry`，企业需要一个能支持所有格式的**通用制品库**。

### 1.1 支持的制品格式

| 格式 | 说明 |
|------|------|
| **Docker** | 容器镜像 |
| **Maven** | Java 依赖 |
| **npm** | Node.js 包 |
| **PyPI** | Python 包 |
| **Go** | Go Modules |
| **Helm** | Kubernetes Charts |
| **Generic** | 任意二进制文件 |

### 1.2 企业级功能

- **统一管理** - 一个平台管理所有制品
- **权限控制** - 精细化的访问控制
- **安全扫描** - 集成 Xray 扫描漏洞
- **高可用** - 支持集群部署
- **复制同步** - 多数据中心同步

## 2. 核心架构：三大仓库类型

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Artifactory                                    │
│                                                                        │
│   ┌─────────────────┐                                                 │
│   │  虚拟仓库        │ ◀──── 开发人员/Jenkins 访问这个                  │
│   │  (Virtual)      │       统一入口                                   │
│   └────────┬────────┘                                                 │
│            │                                                          │
│            │ 聚合                                                      │
│            ▼                                                          │
│   ┌─────────────────┐    ┌─────────────────┐                         │
│   │  本地仓库        │    │  远程仓库        │                         │
│   │  (Local)        │    │  (Remote)       │                         │
│   │                 │    │                 │                         │
│   │  存放内部构建    │    │  代理公网源      │                         │
│   │  产物           │    │  (缓存加速)      │                         │
│   └─────────────────┘    └────────┬────────┘                         │
│                                   │                                   │
│                                   │ 代理                              │
│                                   ▼                                   │
│                          ┌─────────────────┐                         │
│                          │  公网源          │                         │
│                          │  npmjs.org      │                         │
│                          │  maven central  │                         │
│                          │  docker hub     │                         │
│                          └─────────────────┘                         │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.1 本地仓库 (Local Repository)

**用途：** 存放公司内部构建产物

- CI 构建完成后推送到这里
- 存放公司私有的包和镜像
- 通常按项目或团队划分

```bash
# Maven 发布示例
mvn deploy -DaltDeploymentRepository=libs-release::default::https://artifactory.example.com/artifactory/libs-release-local
```

### 2.2 远程仓库 (Remote Repository)

**用途：** 代理公网源，提供缓存

- 配置指向公网源（npmjs, maven central 等）
- 自动缓存下载过的依赖
- 断网时仍可使用缓存

```
远程仓库配置示例：

名称: npm-remote
URL: https://registry.npmjs.org
缓存时间: 7200 秒
```

### 2.3 虚拟仓库 (Virtual Repository)

**用途：** 聚合多个仓库，提供统一入口

- 将 Local + Remote 打包成一个 URL
- 优先级可配置（先查私有库，再查公网缓存）
- 开发人员只需配置一个地址

```
虚拟仓库配置：

名称: npm-virtual
包含仓库:
  1. npm-local (优先级高)
  2. npm-remote (优先级低)
```

## 3. 数据流向

### 3.1 下载 (Pull) 流程

```
开发人员/Jenkins
       │
       │ npm install / mvn install
       ▼
┌─────────────────┐
│  虚拟仓库        │
│  (npm-virtual)  │
└────────┬────────┘
         │
         │ 1. 先查本地仓库
         ▼
┌─────────────────┐    找到 ──▶ 返回
│  本地仓库        │
│  (npm-local)    │
└────────┬────────┘
         │
         │ 2. 未找到，查远程仓库缓存
         ▼
┌─────────────────┐    缓存命中 ──▶ 返回
│  远程仓库        │
│  (npm-remote)   │
└────────┬────────┘
         │
         │ 3. 缓存未命中，从公网下载
         ▼
┌─────────────────┐
│  npmjs.org      │ ──▶ 下载并缓存 ──▶ 返回
└─────────────────┘
```

### 3.2 发布 (Push) 流程

```
Jenkins CI
    │
    │ mvn deploy / npm publish / docker push
    ▼
┌─────────────────┐
│  本地仓库        │
│  (libs-release) │
└────────┬────────┘
         │
         │ 自动同步到虚拟仓库
         ▼
┌─────────────────┐
│  虚拟仓库        │ ──▶ 供 K8s/开发人员拉取
└─────────────────┘
```

## 4. 配置示例

### 4.1 Maven settings.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<settings>
  <servers>
    <server>
      <id>artifactory</id>
      <username>${env.ARTIFACTORY_USER}</username>
      <password>${env.ARTIFACTORY_PASSWORD}</password>
    </server>
  </servers>

  <mirrors>
    <mirror>
      <id>artifactory</id>
      <name>Artifactory Mirror</name>
      <url>https://artifactory.example.com/artifactory/maven-virtual/</url>
      <mirrorOf>*</mirrorOf>
    </mirror>
  </mirrors>
</settings>
```

### 4.2 npm .npmrc

```ini
registry=https://artifactory.example.com/artifactory/api/npm/npm-virtual/
//artifactory.example.com/artifactory/api/npm/npm-virtual/:_authToken=${NPM_TOKEN}
```

### 4.3 Docker daemon.json

```json
{
  "insecure-registries": [],
  "registry-mirrors": ["https://artifactory.example.com"]
}
```

### 4.4 Dockerfile 使用私有镜像

```dockerfile
FROM artifactory.example.com/docker-virtual/node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install --registry=https://artifactory.example.com/artifactory/api/npm/npm-virtual/
COPY . .
CMD ["npm", "start"]
```

## 5. Jenkins 集成

### 5.1 Jenkinsfile 配置

```groovy
pipeline {
    agent any

    environment {
        ARTIFACTORY_URL = 'https://artifactory.example.com/artifactory'
        DOCKER_REGISTRY = 'artifactory.example.com'
    }

    stages {
        stage('Build') {
            steps {
                // 使用 Artifactory 作为 Maven 仓库
                configFileProvider([configFile(fileId: 'maven-settings', variable: 'MAVEN_SETTINGS')]) {
                    sh 'mvn -s $MAVEN_SETTINGS clean package'
                }
            }
        }

        stage('Publish Artifact') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'artifactory-credentials',
                    usernameVariable: 'ARTIFACTORY_USER',
                    passwordVariable: 'ARTIFACTORY_PASSWORD'
                )]) {
                    sh '''
                        mvn deploy \
                            -DaltDeploymentRepository=libs-release::default::${ARTIFACTORY_URL}/libs-release-local
                    '''
                }
            }
        }

        stage('Docker Push') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'artifactory-credentials',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo $DOCKER_PASS | docker login ${DOCKER_REGISTRY} -u $DOCKER_USER --password-stdin
                        docker tag my-app:latest ${DOCKER_REGISTRY}/docker-local/my-app:${BUILD_NUMBER}
                        docker push ${DOCKER_REGISTRY}/docker-local/my-app:${BUILD_NUMBER}
                    '''
                }
            }
        }
    }
}
```

## 6. 优势总结

| 优势 | 说明 |
|------|------|
| **统一入口** | 全公司只配置一个 URL，简化管理 |
| **安全合规** | 支持 Xray 扫描漏洞，精细化权限控制 |
| **速度稳定** | 内网缓存公网依赖，构建速度显著提升 |
| **断网保护** | 公网不可用时，缓存仍可正常使用 |
| **版本追溯** | 所有制品都有完整的版本历史 |
| **合规审计** | 记录所有下载和发布操作 |

## 7. 常见问题

### 7.1 清理策略

配置自动清理策略，避免磁盘空间耗尽：

```
清理策略示例：
- 保留最近 30 天的 SNAPSHOT
- 保留最近 10 个版本的 Release
- 下载次数为 0 且超过 90 天的制品自动删除
```

### 7.2 高可用部署

生产环境建议：
- 使用外部数据库（PostgreSQL/MySQL）
- 使用共享存储（NFS/S3）
- 部署多个节点做负载均衡

### 7.3 Harbor vs Artifactory

| 特性 | Harbor | Artifactory |
|------|--------|-------------|
| **支持格式** | 仅 Docker/Helm | 全格式支持 |
| **开源** | 完全开源 | 社区版功能有限 |
| **成本** | 免费 | 企业版收费 |
| **适用场景** | 中小团队 | 大型企业 |

对于只需要 Docker 镜像管理的场景，Harbor 是很好的免费替代方案。
