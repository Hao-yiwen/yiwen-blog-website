---
title: VS Code 断点调试 npm 脚本与第三方依赖
authors: [yiwen]
tags: [VS Code, Node.js, 调试]
date: 2024-01-16
---

调试复杂脚手架项目时尝试配置了 `VS Code Debugger`，发现其源码映射与断点追踪能力非常强大，即便是编译进 `node_modules` 的第三方包也能精准命中断点。

<!-- truncate -->

## 场景与配置

调试场景：通过 `npm run start:crn` 启动子目录工程，其中某个本地联调的第三方包通过 `tsc` 编译输出到 `node_modules` 中，需要在该包源码/产物中打断点单步跟踪。

对应的 `.vscode/launch.json` 配置如下：

```json title=".vscode/launch.json"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch via NPM",
            "type": "node",
            "request": "launch",
            "runtimeExecutable": "npm",
            "runtimeArgs": ["run-script", "start:crn"],
            "skipFiles": ["<node_internals>/**"],
            "cwd": "${workspaceFolder}/xtaro-demo1",
            "outFiles": ["${workspaceFolder}/xtaro-demo1/node_modules/**/*.js"]
        }
    ]
}
```
