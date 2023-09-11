---
sidebar_position: 1
---

# Npm包

## 可执行Npm包的运行逻辑

在`package.json`文件中，bin字段用于指定一个或多个可执行文件。这些文件通常是项目提供的命令行工具。当用户通过`npm install`或`yarn install`安装该包后，`npm`或`yarn`会在`node_modules/.bin/`目录内为这些可执行文件创建软链接（也称为符号链接）。

例如，如果你有一个名为`my-cli.js`的可执行文件，并希望将其作为命令行工具`my-cli`供用户使用，你可以在`package.json`中进行如下设置：

```json title="package.json"
{
    "name": "my-package",
    "version": "1.0.0",
    "bin": {
        "my-cli": "./path/to/my-cli.js"
    }
}
```

当用户全局安装该包（使用`npm install -g`）或在项目中安装该包（使用`npm install --save`）后，软链接会被创建。如果是全局安装，软链接还会被添加到系统的全局可执行文件路径中（如 `/usr/local/bin`），这样用户就可以在命令行中直接使用my-cli。

:::tip 提示
使用NVM（Node Version Manager）管理Node.js版本时，全局安装的npm包不会存放在系统全局目录（如 `/usr/local/bin`），而是存放在当前激活的`Node.js`版本的对应目录中，例如：`/Users/username/.nvm/versions/node/v14.15.4/bin`。

这种设计允许每个`Node.js`版本拥有自己独立的全局环境，避免了版本切换时可能出现的问题。
:::

可执行文件的查找逻辑是这样的：当你在项目中执行一个命令（例如`taro`）时，系统首先会在项目的`node_modules/.bin`目录下查找对应的可执行文件。如果在该目录下没有找到，系统会继续在全局的`.bin`目录中查找。如果全局目录中也没有找到对应的可执行文件，那么会返回一个错误信息。

这是因为在执行命令时，系统会检查所有在`PATH`环境变量中列出的目录。`nvm`在激活某个`Node.js`版本时，会自动将该版本的`bin`目录添加到`PATH`中，从而使得在这个目录下的可执行文件都可以被直接运行。

## 依赖项不同版本

### 场景1: 在项目中单独依赖

如果`xtaro`依赖了`taro`版本3.5.12，
并且你的项目也依赖了`taro`版本3.6.13，
在这种情况下，通常两个不同版本的taro都会被安装。

`3.6.13` 版本会被安装在项目根目录下的 `node_modules/taro`。
`3.5.12` 版本可能会被安装在 `node_modules/xtaro/node_modules/taro`。

### 场景2: 使用了npm的dedupe功能

使用了`npm dedupe`命令或者在一个较新版本的npm环境（通常`5.x`以上）下，npm会尽量将依赖扁平化，以减少项目大小和提高效率。在这种情况下，`npm`会尝试仅安装一个兼容多个地方的`taro`版本。

这里的“兼容”是基于`semver`（语义版本）规范来判断的。如果`xtaro`和你的项目都能接受一个共同的`taro`版本，`npm`会尝试只安装这一个版本。

注意
依赖冲突和版本管理是复杂的问题，具体情况可能因npm的版本和配置以及`package.json`或`package-lock.json/yarn.lock`的具体内容而有所不同。

总体而言，如果`xtaro`和你的项目都明确要求了特定版本的`taro`，并且这两个版本是不兼容的，那么两个版本都会被安装。否则，`npm`会尝试优化并可能只安装一个共同兼容的版本。

## 请勿滥用post-install

在 npm（或 yarn）中，postinstall 是一个在库安装后自动运行的脚本。如果一个库（或包）在其 package.json 文件中定义了一个 postinstall 脚本，当您通过 npm（或 yarn）安装这个库时，该脚本将自动执行。

:::danger
使用 postinstall 脚本应当谨慎，因为它们有潜在的安全风险。确保你完全信任你所安装的包，以及它们的 postinstall 脚本。

如果你是库的维护者，不要在 postinstall 脚本中执行可能对用户不明确或意外的操作。

对于全局安装的包，postinstall 脚本将以全局范围内运行，这可能需要更高的系统权限。

有时网络或权限问题可能会导致 postinstall 脚本失败，这也会导致整个安装过程失败。

在某些情况下，用户可能使用 --ignore-scripts 标志来禁用 postinstall 脚本，以避免其执行。
:::
