# vscode调试

今天在调试项目的时候想着尝试用一下`vscode debugger`，本来没抱有什么期望，但是在使用后，感觉就是调试能力很强大，终于感觉`vscode`是个强大的`ide`了。

今天调试的任务是`npm run start`,然后在启动的时候有个三方包从其他地方`tsc`到`node_moduels`中，然后我再三方包中打断点调试，这个`launch.json`如下:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch via NPM",
            "type": "node",
            "request": "launch",
            "runtimeExecutable": "npm",
            "runtimeArgs": [
                "run-script",
                "start:crn"
            ],
            "skipFiles": [
                "<node_internals>/**"
            ],
            "cwd": "${workspaceFolder}/xtaro-demo1", // 设置当前工作目录为 xtaro-demo1
            "outFiles": ["${workspaceFolder}/xtaro-demo1/node_modules/**/*.js"]
        }
    ]
}
```