---
sidebar_position: 3
---

# 网关配置

## maven依赖

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-gateway</artifactId>
</dependency>
```

## bootstrap.yml配置

```yml
server:
    port: 63010 # 网关端口
spring:
    cloud:
        gateway:
            #      filter:
            #        strip-prefix:
            #          enabled: true
            routes: # 网关路由配置
                - id: content-api # 路由id，自定义，只要唯一即可
                  # uri: http://127.0.0.1:8081 # 路由的目标地址 http就是固定地址
                  uri: lb://content-api # 路由的目标地址 lb就是负载均衡，后面跟服务名称
                  predicates: # 路由断言，也就是判断请求是否符合路由规则的条件
                      - Path=/content/** # 这个是按照路径匹配，只要以/content/开头就符合要求
                #          filters:
                #            - StripPrefix=1
                - id: system-api
                  # uri: http://127.0.0.1:8081
                  uri: lb://system-api
                  predicates:
                      - Path=/system/**
                #          filters:
                #            - StripPrefix=1
                - id: media-api
                  # uri: http://127.0.0.1:8081
                  uri: lb://media-api
                  predicates:
                      - Path=/media/**
                #          filters:
                #            - StripPrefix=1
                - id: search-service
                  # uri: http://127.0.0.1:8081
                  uri: lb://search
                  predicates:
                      - Path=/search/**
                #          filters:
                #            - StripPrefix=1
                - id: auth-service
                  # uri: http://127.0.0.1:8081
                  uri: lb://auth-service
                  predicates:
                      - Path=/auth/**
                #          filters:
                #            - StripPrefix=1
                - id: checkcode
                  # uri: http://127.0.0.1:8081
                  uri: lb://checkcode
                  predicates:
                      - Path=/checkcode/**
                #          filters:
                #            - StripPrefix=1
                - id: learning-api
                  # uri: http://127.0.0.1:8081
                  uri: lb://learning-api
                  predicates:
                      - Path=/learning/**
                #          filters:
                #            - StripPrefix=1
                - id: orders-api
                  # uri: http://127.0.0.1:8081
                  uri: lb://orders-api
                  predicates:
                      - Path=/orders/**
#          filters:
#            - StripPrefix=1
```

## 直接地址和服务名称

-   使用直接地址（如 http://127.0.0.1:8081）是简单和直接的，但不适用于动态伸缩或微服务环境，因为你必须手动管理服务实例和地址。

-   使用服务名称（如 lb://content-api）则更适用于微服务架构，特别是当服务实例动态改变和需要负载均衡时。

lb:// 前缀表示使用负载均衡（Load Balanced）
