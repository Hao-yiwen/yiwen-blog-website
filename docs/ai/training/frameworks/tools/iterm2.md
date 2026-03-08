---
title: iTerm2 完整使用指南
sidebar_label: iTerm2 指南
date: 2026-01-05
tags: [iterm2, macos, terminal, 工具, ai]
---

# iTerm2 完整使用指南

## 安装

```bash
brew install --cask iterm2
```

---

## 核心快捷键

### 窗口 & 标签页

| 操作 | 快捷键 |
|------|--------|
| 新建窗口 | `Cmd + N` |
| 新建标签页 | `Cmd + T` |
| 关闭标签页/窗口 | `Cmd + W` |
| 切换标签页 | `Cmd + 数字` 或 `Cmd + 左/右` |
| 全屏 | `Cmd + Enter` |

### 分屏

| 操作 | 快捷键 |
|------|--------|
| 垂直分屏（左右） | `Cmd + D` |
| 水平分屏（上下） | `Cmd + Shift + D` |
| 切换分屏 | `Cmd + Option + 方向键` |
| 最大化当前分屏 | `Cmd + Shift + Enter` |

### 搜索 & 历史

| 操作 | 快捷键 |
|------|--------|
| 搜索 | `Cmd + F` |
| 自动补全 | `Cmd + ;` |
| 粘贴历史 | `Cmd + Shift + H` |
| 命令历史 | `Cmd + Shift + ;` |
| 最近目录 | `Cmd + Option + /` |

### 编辑

| 操作 | 快捷键 |
|------|--------|
| 清屏 | `Cmd + K` |
| 清除当前行 | `Ctrl + U` |
| 选中即复制 | 鼠标选中自动复制 |
| 打开 URL/路径 | `Cmd + 点击` |
| 矩形选择 | `Cmd + Option + 拖拽` |

---

## 热键窗口（Hotkey Window）

随时从屏幕顶部滑下的快捷终端：

1. 打开 `Settings > Keys > Hotkey`
2. 勾选 `Create a Dedicated Hotkey Window`
3. 设置快捷键（如 `Option + Space`）

效果：按快捷键，终端从屏幕顶部滑出；再按一次自动隐藏。

---

## Shell Integration

Shell Integration 让 iTerm2 理解你的 shell，解锁高级功能。

### 安装

```bash
# 自动安装（推荐）
# 菜单：iTerm2 > Install Shell Integration

# 或手动安装
curl -L https://iterm2.com/shell_integration/zsh -o ~/.iterm2_shell_integration.zsh
echo 'source ~/.iterm2_shell_integration.zsh' >> ~/.zshrc
```

### 解锁功能

- **命令历史**：记录所有命令，`Cmd + Shift + ;` 快速搜索
- **最近目录**：`Cmd + Option + /` 快速跳转
- **命令状态**：命令失败时左侧标记变红
- **自动补全增强**：命令历史也加入补全
- **时间戳**：显示每条命令执行时间

---

## AI 功能（新）

iTerm2 现在支持 AI 集成，需要先安装 AI Plugin。

### 1. 安装 AI Plugin

1. 下载：https://iterm2.com/ai-plugin.html
2. 解压后将 app 放入 `/Applications`
3. 打开 `Settings > General > AI`
4. 勾选 `Enable generative AI features`
5. 配置 API Key（支持 OpenAI、Anthropic、本地 Ollama 等）

### 2. AI 功能一览

#### Command Generator（命令生成器）

用自然语言描述，AI 生成命令：

- **快捷键**：`Cmd + Y`
- **使用**：输入描述（如 "查找大于 100MB 的文件"），按 `Shift + Enter` 生成
- **执行**：确认后 `Shift + Enter` 运行

#### AI Chat（AI 聊天）

与 AI 对话，可以关联当前终端：

- **快捷键**：`Ctrl + Shift + Cmd + Y`
- **菜单**：`Session > Open AI Chat`
- **权限控制**：可授权 AI 执行命令、查看历史、读写文件等

#### Codecierge（代码助手）

多步骤任务的 AI 助手：

- **打开**：`Toolbelt > Show Toolbelt`，勾选 `Codecierge`
- **使用**：描述任务，AI 给出分步命令
- **特点**：执行每条命令后，AI 会分析输出并给出下一步

#### Explain Output（解释输出）

- **菜单**：`Edit > Explain Output with AI`
- **功能**：AI 解释当前命令输出，添加注释说明

#### AI Composer 补全

在输入命令时实时 AI 补全建议：

- **开启**：`Settings > General > AI > Features`
- **勾选**：`AI Completion in Composer`
- **注意**：默认关闭，因为会发送输入内容给 AI

### 3. 本地 AI（Ollama）配置

不想数据上云？可以用本地模型：

```bash
# 安装 Ollama
brew install ollama

# 下载模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

iTerm2 配置：
- **API URL**：`http://localhost:11434/v1/chat/completions`
- **Model**：`qwen2.5:7b`

---

## 实用技巧

### 触发器（Triggers）

自动响应特定输出：

`Settings > Profiles > Advanced > Triggers`

示例：
- 匹配 `error` 时高亮红色
- 匹配 `BUILD SUCCESS` 时播放提示音

### 即时回放（Instant Replay）

查看终端历史输出：`Cmd + Option + B`

### 密码管理器

`Window > Password Manager`，安全存储常用密码。

### Badge

在终端显示当前信息（如主机名）：

`Settings > Profiles > General > Badge`

示例：`\(session.hostname)`

---

## 推荐配置

### 主题

```bash
# 安装 Dracula 主题
git clone https://github.com/dracula/iterm.git
# 导入：Settings > Profiles > Colors > Color Presets > Import
```

### 字体

推荐使用支持连字的等宽字体：
- **JetBrains Mono**
- **Fira Code**
- **Cascadia Code**

### Oh My Zsh + Starship

```bash
# Oh My Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Starship prompt
brew install starship
echo 'eval "$(starship init zsh)"' >> ~/.zshrc
```

---

## 常见问题

**Q: 分屏后怎么调整大小？**
A: 拖拽分割线，或 `Cmd + Ctrl + 方向键`

**Q: 怎么恢复误关的 session？**
A: `Cmd + Z`（5秒内有效）

**Q: AI 功能没反应？**
A: 检查 AI Plugin 是否安装、API Key 是否正确、网络是否通畅

---

## 相关链接

- 官网：https://iterm2.com
- AI Plugin：https://iterm2.com/ai-plugin.html
- Shell Integration：https://iterm2.com/shell_integration.html
- 文档：https://iterm2.com/documentation.html
