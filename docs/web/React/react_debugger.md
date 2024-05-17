# React@18.2.0源码调试

## 参考文档

[react源码调试](https://juejin.cn/post/7126501202866470949)

## 代码仓库

[react@18.2.0源码](https://github.com/Hao-yiwen/react/tree/18.2.0-hyw-debug)

[cra项目:react-debugger](https://github.com/Hao-yiwen/react-debugger)

```json title="vscode debugger"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch chrome",
            "request": "launch",
            "type": "chrome",
            "url": "http://localhost:3000",
            "webRoot": "${workspaceFolder}/react-debugger",
        }
    ]
}

```

## 使用流程

```bash
# 启动vscode debugger

# react-debugger启动项目
yarn start
```