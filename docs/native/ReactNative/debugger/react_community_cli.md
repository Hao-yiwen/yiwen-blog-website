# @react-native-community/cli调试

1. [克隆仓库](https://github.com/react-native-community/cli)

2. [修复小bug](https://github.com/react-native-community/cli/issues/2442#issuecomment-2205939960)

3. yarn

4. 运行命令: `node --inspect-brk /Users/haoyiwen/Documents/rn/cli/packages/cli/build/bin.js bundle --platform android --dev false --entry-file index.js --bundle-output build/rnDemo0742.jsbundle --assets-dest build`

5. vscode attach:

```json
{
    "name": "Attach",
    "port": 9229,
    "request": "attach",
    "skipFiles": ["<node_internals>/**"],
    "type": "node"
}
```
