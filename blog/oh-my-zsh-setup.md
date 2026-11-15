---
title: Oh My Zsh 命令自动补全与语法高亮配置
authors: [yiwen]
tags: [macOS, Zsh, 终端]
date: 2023-09-12
---

记录在 macOS / Linux 终端中安装 `Oh My Zsh` 并集成 `zsh-autosuggestions`（历史命令补全）与 `zsh-syntax-highlighting`（命令语法高亮）两个必备插件的步骤。

<!-- truncate -->

## 1. 安装 Oh My Zsh

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

安装完成后可按需编辑 `~/.zshrc` 配置主题：

```bash
ZSH_THEME="robbyrussell"
```

## 2. 配置命令自动提示（zsh-autosuggestions）

```bash
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

## 3. 配置命令语法高亮（zsh-syntax-highlighting）

参考[官方安装文档](https://github.com/zsh-users/zsh-syntax-highlighting/blob/master/INSTALL.md)：

```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

## 4. 启用插件并生效

打开 `~/.zshrc`，在 `plugins` 列表中加入上述两个插件：

```bash title="~/.zshrc"
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
)
```

重新加载配置文件使设置立即生效：

```bash
source ~/.zshrc
```
