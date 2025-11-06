---
title: 路由守卫
sidebar_label: 路由守卫
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 路由守卫

路由守卫（Route Guard）是前端路由管理中的一个概念，主要用于控制用户对特定路由的访问权限。它在单页应用（SPA）如Angular、React、Vue.js等框架中尤为常见，用于在路由级别处理认证和授权。

## 路由守卫的作用

1. 访问控制：防止未经授权的用户访问某些路由。例如，阻止未登录用户访问需要认证的页面。
2. 重定向：根据用户的登录状态或权限，自动重定向到不同的路由。
3. 数据预加载：在路由激活之前预先加载所需数据。
4. 条件渲染：基于特定条件（如用户角色、功能开关）决定是否渲染某个路由。

## 示例：React路由守卫

```js
// 在React中，路由守卫可以通过创建一个高阶组件来实现：
import React from 'react';
import { Redirect, Route } from 'react-router-dom';

const ProtectedRoute = ({ component: Component, ...rest }) => {
    const isAuthenticated = // 逻辑判断用户是否认证
    return (
        <Route
            {...rest}
            render={props =>
                isAuthenticated ? <Component {...props} /> : <Redirect to="/login" />
            }
        />
    );
};

// 在路由配置中使用
<ProtectedRoute path="/protected" component={ProtectedComponent} />
```

## 示例：Umi路由守卫

在`app.ts`中进行如下配置
```js
export function onRouteChange({ location, routes, action }) {
  const token = localStorage.getItem('userToken');
  const isLogin = !!token; // 根据实际情况判断登录状态
  const isLoginPage = location.pathname === '/login';

  // 如果用户未登录且不在登录页面，则重定向到登录页面
  if (!isLogin && !isLoginPage) {
    location.pathname = '/login';
  }
}
```