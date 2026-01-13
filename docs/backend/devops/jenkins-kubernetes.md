---
title: Jenkins 与 Kubernetes 的深度集成
sidebar_position: 3
tags: [jenkins, kubernetes, k8s, gitops, argocd]
---

# Jenkins 与 Kubernetes 的深度集成

## 1. 动态扩缩容 (Dynamic Provisioning)

传统 Jenkins 需要维护长驻的虚拟机 Agent，资源利用率低。与 Kubernetes 集成后，实现**动态扩缩容**：

```
有任务时 → 申请 K8s Pod → 运行任务 → 任务结束 → 销毁 Pod
```

### 1.1 优势

| 特性 | 说明 |
|------|------|
| **节省资源** | 无任务时不占用资源 |
| **环境隔离** | 每次构建都是全新环境，用完即焚 |
| **无限并发** | 理论上可以同时运行任意数量的构建 |
| **版本一致** | 通过镜像保证构建环境完全一致 |

### 1.2 架构图

```
┌────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│                                                             │
│  ┌─────────────┐                                            │
│  │   Jenkins   │                                            │
│  │   Master    │                                            │
│  └──────┬──────┘                                            │
│         │ 创建 Pod                                          │
│         ▼                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Build Pod  │  │  Build Pod  │  │  Build Pod  │  ...    │
│  │   (Maven)   │  │   (Node)    │  │   (Go)      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          │                                  │
│                    任务完成后自动销毁                         │
└────────────────────────────────────────────────────────────┘
```

## 2. Pod Template (Pod 模板)

Pod Template 告诉 Kubernetes 你需要什么样的构建环境。

### 2.1 基础示例

```groovy
pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
metadata:
  name: build-pod
spec:
  containers:
  - name: maven
    image: maven:3.8-jdk-11
    command: ['cat']
    tty: true
    volumeMounts:
    - name: maven-cache
      mountPath: /root/.m2
  volumes:
  - name: maven-cache
    persistentVolumeClaim:
      claimName: maven-cache-pvc
"""
        }
    }

    stages {
        stage('Build') {
            steps {
                container('maven') {
                    sh 'mvn clean package'
                }
            }
        }
    }
}
```

### 2.2 多容器 Pod

一个 Pod 中可以包含多个容器，满足不同阶段的需求：

```groovy
pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: maven
    image: maven:3.8-jdk-11
    command: ['cat']
    tty: true

  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
    env:
    - name: DOCKER_TLS_CERTDIR
      value: ""

  - name: kubectl
    image: bitnami/kubectl:latest
    command: ['cat']
    tty: true
"""
        }
    }

    stages {
        stage('Build') {
            steps {
                container('maven') {
                    sh 'mvn clean package -DskipTests'
                }
            }
        }

        stage('Docker Build') {
            steps {
                container('docker') {
                    sh 'docker build -t my-app:v1.0 .'
                }
            }
        }

        stage('Deploy') {
            steps {
                container('kubectl') {
                    sh 'kubectl apply -f k8s/'
                }
            }
        }
    }
}
```

## 3. 现代 CI/CD 架构：Build - Ship - Run

在现代架构中，Jenkins 不再直接负责部署，而是转向 **GitOps** 模式。

### 3.1 架构流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CI/CD 完整流程                                 │
└─────────────────────────────────────────────────────────────────────────┘

       CI (Jenkins)                    Registry              CD (Argo CD)
┌─────────────────────┐          ┌─────────────────┐    ┌─────────────────┐
│                     │          │                 │    │                 │
│  1. 拉取代码        │          │                 │    │  5. 监听 Git    │
│  2. 编译构建        │  Push    │     Harbor      │    │  6. 发现变化    │
│  3. 运行测试        │ ──────▶  │   (镜像仓库)    │    │  7. 同步 K8s    │
│  4. 打包镜像        │          │                 │    │                 │
│                     │          └────────┬────────┘    └────────┬────────┘
└──────────┬──────────┘                   │                      │
           │                              │      Pull            │
           │ 更新版本号                    │  ◀────────────────────
           ▼                              │
┌─────────────────────┐                   │
│   Config Git Repo   │                   ▼
│   (K8s YAML 文件)   │          ┌─────────────────┐
│   - deployment.yaml │          │   Kubernetes    │
│   - service.yaml    │          │     Cluster     │
└─────────────────────┘          └─────────────────┘
```

### 3.2 优势

| 优势 | 说明 |
|------|------|
| **安全** | Jenkins 不需要 K8s 集群的写权限 |
| **可审计** | 所有变更通过 Git 记录，有完整历史 |
| **可回滚** | Git revert 即可回滚到任意版本 |
| **状态可视** | Argo CD 提供清晰的同步状态展示 |

### 3.3 Jenkins 端配置

Jenkins 只负责到"推送镜像 + 更新 Config 仓库"为止：

```groovy
pipeline {
    agent any

    environment {
        REGISTRY = 'harbor.example.com'
        IMAGE_NAME = 'my-app'
        CONFIG_REPO = 'git@github.com:org/k8s-config.git'
    }

    stages {
        stage('Build & Push Image') {
            steps {
                sh """
                    docker build -t ${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} .
                    docker push ${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                """
            }
        }

        stage('Update Config Repo') {
            steps {
                withCredentials([sshUserPrivateKey(
                    credentialsId: 'git-ssh-key',
                    keyFileVariable: 'SSH_KEY'
                )]) {
                    sh """
                        # 克隆配置仓库
                        git clone ${CONFIG_REPO} config-repo
                        cd config-repo

                        # 更新镜像版本
                        sed -i 's|image: ${REGISTRY}/${IMAGE_NAME}:.*|image: ${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}|' \
                            deployments/my-app/deployment.yaml

                        # 提交变更
                        git config user.email "jenkins@example.com"
                        git config user.name "Jenkins"
                        git add .
                        git commit -m "Update ${IMAGE_NAME} to version ${BUILD_NUMBER}"
                        git push origin main
                    """
                }
            }
        }
    }
}
```

## 4. Kubernetes 插件配置

### 4.1 Jenkins 系统配置

在 Jenkins 管理界面配置 Kubernetes 云：

1. **系统管理** → **节点管理** → **配置云**
2. 添加 **Kubernetes** 云
3. 配置项：
   - Kubernetes URL: `https://kubernetes.default.svc`
   - Kubernetes Namespace: `jenkins`
   - Jenkins URL: `http://jenkins.jenkins.svc:8080`
   - Jenkins Tunnel: `jenkins-agent.jenkins.svc:50000`

### 4.2 ServiceAccount 配置

Jenkins 需要适当的权限来创建 Pod：

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jenkins
  namespace: jenkins
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: jenkins-admin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin  # 生产环境应使用更细粒度的权限
subjects:
- kind: ServiceAccount
  name: jenkins
  namespace: jenkins
```

## 5. 最佳实践

### 5.1 镜像缓存

使用 PVC 缓存依赖，加速构建：

```yaml
volumes:
- name: maven-cache
  persistentVolumeClaim:
    claimName: maven-cache-pvc
- name: npm-cache
  persistentVolumeClaim:
    claimName: npm-cache-pvc
```

### 5.2 资源限制

为构建 Pod 设置资源限制，防止影响其他服务：

```yaml
containers:
- name: maven
  image: maven:3.8-jdk-11
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
```

### 5.3 镜像拉取策略

```yaml
containers:
- name: build
  image: my-registry/build-image:latest
  imagePullPolicy: Always  # 确保使用最新镜像
```
