# 常用rollup插件

[官方插件仓库](https://github.com/rollup/plugins)

## 常用插件

```js
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';
import json from '@rollup/plugin-json';
import terser from '@rollup/plugin-terser'

export default [
  {
    input: 'src/test.ts',
    output: {
      file: 'dist/test.js',
      format: 'cjs',
    },
    plugins: [
      resolve(), // 该插件告诉 Rollup 如何查找外部模块
      commonjs(), // 该插件将 CommonJS 模块转换为 ES2015 供 Rollup 处理
      typescript(), // 该插件将 TypeScript 转换为 JavaScript
      json(), // 该插件将 JSON 转换为 ES6 模块
      terser(), // 该插件将压缩包中的代码进行压缩
    ]
  },
]
```