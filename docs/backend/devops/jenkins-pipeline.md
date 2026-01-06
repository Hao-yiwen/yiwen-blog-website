---
title: Pipeline 开发与执行机制
sidebar_position: 2
tags: [jenkins, pipeline, groovy, ci]
---

# Pipeline 开发与执行机制

## 1. 插件 vs 工具

理解 Jenkins 中插件和工具的区别至关重要：

| 类型 | 角色 | 示例 |
|------|------|------|
| **插件 (Plugins)** | "指挥官"，负责逻辑控制 | Git 插件、Pipeline 插件、Docker Pipeline 插件 |
| **工具 (Tools)** | "干苦力的"，实际执行命令 | `mvn`, `npm`, `docker`, `kubectl` |

**关键点：** 必须确保 Jenkins 运行的节点（或镜像）里安装了这些工具，否则会报 `command not found`。

```
插件：知道"怎么做" (HOW)
工具：实际"去做"   (DO)
```

## 2. Jenkinsfile 模板 (声明式流水线)

声明式流水线是最主流的写法，结构清晰，易于维护。

### 2.1 基础模板

```groovy
pipeline {
    agent any // 或指定 kubernetes

    options {
        ansiColor('xterm')      // 开启彩色日志
        timestamps()            // 开启时间戳
        timeout(time: 30, unit: 'MINUTES') // 超时设置
        disableConcurrentBuilds()          // 禁止并发构建
    }

    environment {
        APP_NAME = 'my-app'
        REGISTRY = 'harbor.example.com'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build & Test') {
            steps {
                sh 'echo "开始编译..."'
                sh 'mvn clean package -DskipTests=false'
            }
        }

        stage('Docker Build') {
            steps {
                sh "docker build -t ${REGISTRY}/${APP_NAME}:${BUILD_NUMBER} ."
            }
        }

        stage('Push Image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'harbor-credentials',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo $DOCKER_PASS | docker login ${REGISTRY} -u $DOCKER_USER --password-stdin
                        docker push ${REGISTRY}/${APP_NAME}:${BUILD_NUMBER}
                    '''
                }
            }
        }
    }

    post {
        success {
            echo '构建成功！'
        }
        failure {
            echo '构建失败！'
        }
        always {
            cleanWs() // 清理工作空间
        }
    }
}
```

### 2.2 带参数的流水线

```groovy
pipeline {
    agent any

    parameters {
        string(name: 'BRANCH', defaultValue: 'main', description: '要构建的分支')
        choice(name: 'ENV', choices: ['dev', 'staging', 'prod'], description: '部署环境')
        booleanParam(name: 'SKIP_TESTS', defaultValue: false, description: '是否跳过测试')
    }

    stages {
        stage('Info') {
            steps {
                echo "构建分支: ${params.BRANCH}"
                echo "部署环境: ${params.ENV}"
            }
        }

        stage('Test') {
            when {
                expression { return !params.SKIP_TESTS }
            }
            steps {
                sh 'mvn test'
            }
        }
    }
}
```

## 3. 触发方式

### 3.1 Webhook (推荐)

Git 提交代码 → 实时通知 Jenkins → 立即构建。

```groovy
pipeline {
    triggers {
        // GitHub Webhook
        githubPush()

        // GitLab Webhook
        gitlab(triggerOnPush: true, triggerOnMergeRequest: true)
    }
    // ...
}
```

**配置步骤：**
1. 在 Jenkins 中配置 Git 仓库地址
2. 在 Git 平台配置 Webhook URL: `http://jenkins-url/github-webhook/`
3. 选择触发事件（Push、Pull Request 等）

### 3.2 Cron 定时任务

```groovy
pipeline {
    triggers {
        // 每天凌晨 2 点构建
        cron('0 2 * * *')

        // 工作日每隔 2 小时构建
        cron('0 */2 * * 1-5')
    }
    // ...
}
```

### 3.3 轮询 SCM (不推荐)

```groovy
pipeline {
    triggers {
        // 每 5 分钟检查一次代码变更
        pollSCM('H/5 * * * *')
    }
    // ...
}
```

:::warning 性能警告
轮询方式效率低下，会对 Git 服务器造成压力，不推荐在生产环境使用。
:::

## 4. 常用 Pipeline 语法

### 4.1 条件执行

```groovy
stage('Deploy to Prod') {
    when {
        branch 'main'           // 只在 main 分支执行
        environment name: 'ENV', value: 'prod'  // 环境变量条件
    }
    steps {
        sh './deploy.sh'
    }
}
```

### 4.2 并行执行

```groovy
stage('Parallel Tests') {
    parallel {
        stage('Unit Tests') {
            steps {
                sh 'mvn test'
            }
        }
        stage('Integration Tests') {
            steps {
                sh 'mvn verify'
            }
        }
        stage('Code Analysis') {
            steps {
                sh 'mvn sonar:sonar'
            }
        }
    }
}
```

### 4.3 错误处理

```groovy
stage('Deploy') {
    steps {
        script {
            try {
                sh './deploy.sh'
            } catch (Exception e) {
                echo "部署失败: ${e.message}"
                currentBuild.result = 'UNSTABLE'
            }
        }
    }
}
```

### 4.4 输入确认

```groovy
stage('Deploy to Production') {
    steps {
        input message: '确认部署到生产环境？', ok: '部署'
        sh './deploy-prod.sh'
    }
}
```

## 5. 最佳实践

### 5.1 Jenkinsfile 存放位置

将 `Jenkinsfile` 放在代码仓库根目录，实现 **Pipeline as Code**：

```
my-project/
├── src/
├── Dockerfile
├── Jenkinsfile      # 流水线定义
└── README.md
```

### 5.2 共享库

对于多个项目共用的逻辑，使用 Shared Library：

```groovy
// vars/buildJavaApp.groovy
def call(Map config) {
    pipeline {
        agent any
        stages {
            stage('Build') {
                steps {
                    sh "mvn clean package -P${config.profile}"
                }
            }
        }
    }
}
```

使用方式：

```groovy
@Library('my-shared-library') _
buildJavaApp(profile: 'production')
```

### 5.3 凭据管理

永远不要在 Jenkinsfile 中硬编码密码：

```groovy
// 正确做法
withCredentials([string(credentialsId: 'api-token', variable: 'TOKEN')]) {
    sh 'curl -H "Authorization: Bearer $TOKEN" https://api.example.com'
}

// 错误做法 - 永远不要这样做！
// sh 'curl -H "Authorization: Bearer abc123" https://api.example.com'
```
