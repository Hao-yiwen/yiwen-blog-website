# metro

Metro 是 React Native 的 JavaScript bundler，它负责将 JavaScript 代码打包并提供开发和调试工具。Metro 的主要功能包括模块解析、代码转换和优化，以及热重载（Hot Reloading）和快速刷新（Fast Refresh）。以下是 Metro 的主要组成部分和一个完整的 demo。

## 文档

[metro配置](https://metrobundler.dev/docs/configuration#resolver-options)

## 主要组成部分

1. Resolver（模块解析器）：

-   负责解析 JavaScript 模块的依赖关系。
-   它处理文件扩展名、别名、和自定义解析规则。

2. Transformer（代码转换器）：

-   使用 Babel 将 ES6+ 代码转换为浏览器或运行环境可以理解的 ES5 代码。
-   支持自定义 Babel 插件和预设。

3. Bundler（打包器）：

-   将所有模块及其依赖关系打包成一个或多个 bundle 文件。
-   支持分包（split bundling）以优化加载性能。

4. Server（开发服务器）：

-   提供开发服务器功能，支持实时重新加载（Live Reloading）和快速刷新（Fast Refresh）。
-   用于在开发环境中提供源映射（source maps）和调试功能。

5. Serializer（序列化器）：

-   负责将模块代码序列化成一个 bundle 文件。
-   支持自定义序列化逻辑以优化输出文件。
