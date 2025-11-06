---
title: 契约系统
sidebar_label: 契约系统
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 契约系统

契约系统（Contract Testing）是一种软件测试方法，主要用于验证服务间的交互是否符合事先定义的“契约”。这种方法在微服务架构中尤为重要，因为在这种架构中，服务之间频繁地相互通信。

## 契约系统的核心概念

-   契约：契约是服务之间交互方式的一个正式描述。它定义了服务如何请求另一个服务的API，包括请求的结构、预期的响应以及可能的错误处理方式。
-   消费者：消费者是调用另一个服务（提供者）API的服务。
-   提供者：提供者是提供API供消费者调用的服务。
-   契约测试：这是一种测试方法，用于验证提供者服务是否满足消费者服务的契约要求。

## 契约测试的优势

-   减少集成问题：通过在部署前验证服务间的交互，契约测试有助于减少生产环境中的集成问题。
-   独立部署：服务可以独立于其他服务进行开发和部署，只要它们遵守定义的契约。
-   快速反馈：契约测试提供了关于服务间交互的快速反馈，有助于及时发现并解决问题。
-   文档作用：契约本身可以作为服务间交互的文档，有助于新团队成员理解系统。

## 示例场景
假设有两个服务：

消费者服务：需要从提供者服务获取用户信息。
提供者服务：提供一个API端点来返回用户信息。

## 使用Pact的契约测试（Java）

### 消费者端
1. 添加依赖：在消费者服务的pom.xml中添加Pact依赖。
```xml
<dependency>
  <groupId>au.com.dius</groupId>
  <artifactId>pact-jvm-consumer-junit_2.12</artifactId>
  <version>最新版本</version>
  <scope>test</scope>
</dependency>
```
2. 编写消费者测试：创建一个测试类来定义消费者期望的行为。
```java
import au.com.dius.pact.consumer.dsl.PactDslWithProvider;
import au.com.dius.pact.consumer.junit.Pact;
import au.com.dius.pact.consumer.junit.PactProviderRule;
import au.com.dius.pact.consumer.junit.PactVerification;
import au.com.dius.pact.model.RequestResponsePact;
import org.junit.Rule;
import org.junit.Test;

public class ConsumerPactTest {

  @Rule
  public PactProviderRule mockProvider = new PactProviderRule("ProviderService", this);

  @Pact(consumer = "ConsumerService")
  public RequestResponsePact createPact(PactDslWithProvider builder) {
    return builder
      .given("provider has user information")
      .uponReceiving("a request for user information")
      .path("/user")
      .method("GET")
      .willRespondWith()
      .status(200)
      .body("{\"id\": 1, \"name\": \"John Doe\"}")
      .toPact();
  }

  @Test
  @PactVerification("ProviderService")
  public void testGetUserInformation() {
    // 这里是调用提供者服务的代码
  }
}
```
### 服务端
1. 添加依赖：在提供者服务的pom.xml中添加Pact依赖。
```xml
<dependency>
  <groupId>au.com.dius</groupId>
  <artifactId>pact-jvm-provider-junit_2.12</artifactId>
  <version>最新版本</version>
  <scope>test</scope>
</dependency>
```
2. 编写提供者测试：创建一个测试类来验证提供者是否满足契约。
```java
import au.com.dius.pact.provider.junit.Provider;
import au.com.dius.pact.provider.junit.loader.PactFolder;
import au.com.dius.pact.provider.junit.target.HttpTarget;
import au.com.dius.pact.provider.junit.target.TestTarget;
import au.com.dius.pact.provider.junit5.PactVerification;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestTemplate;

@Provider("ProviderService")
@PactFolder("pacts")
public class ProviderPactTest {

  @TestTarget
  public final HttpTarget target = new HttpTarget(8080); // 假设提供者服务运行在8080端口

  @TestTemplate
  @PactVerification("ProviderService")
  public void verifyPact() {
    // 这里可以编写一些验证逻辑
  }
}
```