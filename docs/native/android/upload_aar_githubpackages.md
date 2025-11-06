---
title: 从githubpackages上传或下载aar
sidebar_label: 从githubpackages上传或下载aar
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 从githubpackages上传或下载aar


## 背景
今天晚上本来想测试一下如何将aar包上传到maven center和从maven center拉取自己的aar。但是在看过了maven center复杂的包要求后(尤其是gpcg配置后)，我果断放弃了在maven center上传代码，我选择githubpackages作为我的第一个aar包上传托管仓库。

## 文档

[githubpackages](https://docs.github.com/zh/packages/working-with-a-github-packages-registry/working-with-the-gradle-registry)

[maven center](https://central.sonatype.org/publish/requirements/#by-code-hosting-services)

## 上传时build.gradle.kts配置

```kt
publishing {
    publications {
        mavenJava(MavenPublication) {
            groupId = 'com.example'
            artifactId = 'chapter03'
            version = '0.0.1'

            // 在 Groovy 中设置 artifact 的方式
            artifact "$buildDir/outputs/aar/chapter03-release.aar"
        }
    }
    repositories {
        maven {
            name = 'GitHubPackages'
            url = uri('https://maven.pkg.github.com/Hao-yiwen/android-study')

            credentials {
                username = project.findProperty('gpr.user') ?: System.getenv('USERNAME')
                password = project.findProperty('gpr.key') ?: System.getenv('TOKEN')
            }
        }
    }
}
```

执行命令如下
```bash
# 先打包release aar包
./gradlew assembleRelease

# 上传aar包
./gradlew publish
```

## 配置仓库源

:::danger
在从githubpackages中拉取aar包的时候有一个小插曲，我一直配置的是`pluginManagement`，从而始终无法正确拉取aar包，后面发现是要配置下面的`dependencyResolutionManagement`(从gradle6.8开始都要在settings.gradle.kts配置)
:::

```kt
// 配置这个不生效，这个是用来解析插件的
pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

// 配置这个
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        maven {
            url = uri("https://maven.pkg.github.com/Hao-yiwen/android-study")
            credentials {
                username = System.getenv("USERNAME") ?: ""
                password = System.getenv("TOKEN") ?: ""
            }
            content {
                includeGroupByRegex("com\\.example.*")
            }
        }
        google()
        mavenCentral()
    }
}
```