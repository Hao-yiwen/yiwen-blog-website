import pnpm_store from "@site/static/img/pnpm_store.png";

# pnpm,yarn和npm

npm（Node Package Manager）、yarn 和 pnpm 都是 JavaScript 生态系统中流行的包管理工具，用于自动化安装、配置和管理项目依赖。尽管它们的基本目的相同，但它们在性能、特性和工作方式上有一些关键的差异

## npm
- 历史与普及度：npm 是最早的包管理器之一，随 Node.js 一起发布，因此在 JavaScript 社区中非常普及。
- 工作方式：npm 从 npm 注册表下载依赖，并在每个项目的 node_modules 文件夹中创建依赖的副本。
- 性能：传统上，npm 的性能（尤其是在安装依赖时）不如 yarn 和 pnpm 快，尽管近年来 npm 做了很多性能优化。
- 锁文件：npm 使用 package-lock.json 文件来锁定依赖的版本，确保不同环境下的一致性。

## yarn
- 历史与普及度：yarn 是由 Facebook 开发的，作为 npm 的替代品，提供更快的依赖安装速度和更好的性能。
- 工作方式：yarn 同样将依赖下载到 node_modules 文件夹，但通过使用缓存和并行下载优化性能。
- 性能：yarn 在性能方面通常优于 npm，尤其是在安装大量依赖的项目时。
- 锁文件：yarn 使用 yarn.lock 文件来锁定依赖版本。
- 特性：yarn 提供了一些 npm 没有的特性，例如 yarn workspaces 用于管理多包项目。

## pnpm
- 历史与普及度：pnpm 相对较新，但因其高效的存储方式和快速的性能而在社区中逐渐流行。
- 工作方式：pnpm 使用硬链接和符号链接来节省磁盘空间，它将下载的包保存在单个位置，并在项目中创建到这些文件的链接。
- 性能：由于其独特的存储方法，pnpm 在磁盘空间利用和安装速度上通常优于 npm 和 yarn。
- 锁文件：pnpm 使用 pnpm-lock.yaml 文件来锁定依赖版本。
- 特性：pnpm 支持严格的依赖树，可以避免子依赖中意外的包版本更新。
- 特点：pnpm使用中央存储办法，所以大多数依赖只需要第一次安装，后续无需安装，并且有效节省硬盘空间。

## Pnpm详细介绍

- 首先pnpm会将依赖下载存储到中央仓库(`/Library/pnpm/store`)中，然后根据项目需要将依赖硬链接到项目的`node_modules/.pnpm`文件夹中，然后在`node_modules`外层都是`.pnpm`文件夹中内容的软连接。`.pnpm`项目中的依赖嵌套内容都是通过软连接连接到对应的文件夹中。

<img src={pnpm_store} width={400} />

- `pnpm`在依赖关系的解决中和`yarn`和`npm`不太相同，如果`pnpm`发现存在依赖相互嵌套问题，都会在`.pnpm`中使用软连接解决，从而优化安装速率和空间，但是这种方式有很多框架不支持，例如`react-native`,如果想要实现传统的`yarn`和`npm`的依赖嵌套方式，需要在`.npmrc`配置`node-linker=hoisted`来指定依赖安装方式。(当您在 .npmrc 配置文件中设置 node-linker=hoisted 时，pnpm 会尽量将依赖项提升到 node_modules 的顶层目录，类似于 npm 或 yarn 的行为。这种提升机制减少了软链接的数量，并尽可能创建一个更扁平化的 node_modules 结构。)