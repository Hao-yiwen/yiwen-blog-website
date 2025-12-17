---
sidebar_position: 2
---

import nacos_setting from "@site/static/img/nacos_setting.png";

# nacos配置

## 架构图

<img src={nacos_setting} style={{width: 700}} />

## pom依赖引入

```xml
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
</dependency>
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
</dependency>
```

## 本地bootstarp.yml基本配置

```yaml title="本地配置"
spring:
    application:
        name: content-api # 服务名
    cloud:
        nacos:
            server-addr: 127.0.0.1:8848
            discovery: # 服务注册
                namespace: dev402
                group: xuecheng-plus-project
            config: # 配置文件相关配置
                namespace: dev402
                group: xuecheng-plus-project
                file-extension: yaml
                refresh-enabled: true
    profiles:
        active: dev # 环境名
```

## nacos远程配置

根据本地配置的规则新建远程配置，例如上述本地配置需要在命名空间`dev402`中创建以下远程配置。

`data Id`: `content-api-dev.yaml`

`group`: `xuecheng-plus-project`

```yaml title="content-api-dev.yaml"
server:
    servlet:
        context-path: /content
    port: 63040
```

## 数据库扩展配置

想必大家发现了，在上述的配置中其实不存在数据库连接，为什么那，因为数据库连接属于`service`层，在另一个配置中，那么我们如何在让一个服务去获得多个配置那，`nacos`拥有扩展配置模块，可轻松解决上述问题。

```yaml title="本地配置"
spring:
    application:
        # ...省略
    cloud:
        nacos:
            # ...省略
            discovery:# 服务注册
                # ...省略
            config: # 配置文件相关配置
                # ...省略
                extension-configs: # 扩展配置
                    - data-id: content-service-${spring.profiles.active}.yaml
                      group: xuecheng-plus-project
                      refresh: true
```

```yaml title="content-service-dev.yaml"
spring:
    datasource:
        driver-class-name: com.mysql.cj.jdbc.Driver
        url: jdbc:mysql://localhost:3306/xc402_content?serverTimezone=UTC&userUnicode=true&useSSL=false&
        username: root
        password: hyw650022
```

## swagger公共配置

```yaml title="本地配置"
spring:
    application:
        # ...省略
    cloud:
        nacos:
            # ...省略
            discovery:# 服务注册
                # ...省略
            config: # 配置文件相关配置
                # ...省略
                extension-configs:# 扩展配置
                    # ...省略
                shared-configs:
                    - data-id: swagger-${spring.profiles.active}.yaml
                      group: xuecheng-plus-common
                      refresh: true
```

```yaml title="swagger-dev.yaml"
# swagger 文档配置
swagger:
    title: '学成在线项目接口文档'
    description: '学成在线项目接口文档'
    base-package: com.xuecheng
    enabled: true
    version: 1.0.0
```

## logger公共配置

```yaml title="本地配置"
spring:
  application:
    # ...省略
  cloud:
    nacos:
      # ...省略
      discovery: # 服务注册
        # ...省略
      config: # 配置文件相关配置
        # ...省略
        extension-configs: # 扩展配置
          # ...省略
        shared-configs:
            # ...省略
            - data-id: logging-${spring.profiles.active}.yaml
            group: xuecheng-plus-common
            refresh: true
```

```yaml title="logging-dev.yaml"
# 日志文件配置路径
logging:
    config: classpath:log4j2-dev.xml
```

## 最终本地nacos配置

```yaml
#微服务配置
spring:
    application:
        name: content-api # 服务名
    cloud:
        nacos:
            server-addr: 127.0.0.1:8848
            discovery: # 服务注册
                namespace: dev402
                group: xuecheng-plus-project
            config: # 配置文件相关配置
                namespace: dev402
                group: xuecheng-plus-project
                file-extension: yaml
                refresh-enabled: true
                extension-configs:
                    - data-id: content-service-${spring.profiles.active}.yaml
                      group: xuecheng-plus-project
                      refresh: true
                shared-configs:
                    - data-id: swagger-${spring.profiles.active}.yaml
                      group: xuecheng-plus-common
                      refresh: true
                    - data-id: logging-${spring.profiles.active}.yaml
                      group: xuecheng-plus-common
                      refresh: true
    profiles:
        active: dev # 环境名
```

## 配置优先级

项目应用名配置文件 > 扩展配置文件 > 共享配置文件 > 本地配置文件

:::note
但是很多时候本地需要启动多个服务，这个时候修改`nacos`配置是不合理的，所以可以通过`vm option`方式传参。

```bash
-Dserver.port=63041
```

:::
