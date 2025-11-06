---
title: springboot3.2设置跨域
sidebar_label: springboot3.2设置跨域
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# springboot3.2设置跨域

## 未开启 Spring Security

配置 Spring MVC 的跨域资源共享设置就可以。

```java
@Override
public void addCorsMappings(CorsRegistry registry) {
    registry.addMapping("/**") // 允许跨域的路径
            .allowedOrigins("*") // 允许跨域的来源
            .allowedMethods("GET", "POST", "PUT", "DELETE") // 允许的方法
            .allowedHeaders("*"); // 允许的头信息
}
```

## 开启 Spring Security

在 Spring Boot 应用中，跨域资源共享（CORS）设置需要在两个地方进行配置：Spring MVC 和 Spring Security，这是因为它们处理请求的方式不同。

Spring MVC CORS 配置：通过 addCorsMappings 方法设置，这影响了 Spring MVC 层面的跨域请求处理。它定义了哪些跨域请求被允许，包括允许的源、方法和头信息。

Spring Security CORS 配置：在 securityFilterChain 方法中，您需要告知 Spring Security 允许跨域请求。Spring Security 有自己的跨域请求处理机制，如果不在这里配置，即使 Spring MVC 允许了跨域请求，Spring Security 可能仍会阻止这些请求。

当这两部分配置都正确设置时，您的应用才能正确地处理跨域请求。这是因为 Spring Security 在请求达到 Spring MVC 之前就会先进行拦截，所以两者的 CORS 配置都需要正确设置。

```java
 @Bean
public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
    http.csrf((csrf) -> csrf.disable())
            .authorizeHttpRequests((authz) -> authz
                    .requestMatchers("/**").permitAll()
                    .anyRequest().authenticated()).httpBasic(httpbase -> httpbase.disable());
    return http.build();
}

@Override
public void addCorsMappings(CorsRegistry registry) {
    registry.addMapping("/**") // 允许跨域的路径
            .allowedOrigins("*") // 允许跨域的来源
            .allowedMethods("GET", "POST", "PUT", "DELETE") // 允许的方法
            .allowedHeaders("*"); // 允许的头信息
}
```