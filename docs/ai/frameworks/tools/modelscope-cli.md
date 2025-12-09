---
title: ModelScope CLI 使用指南
sidebar_label: ModelScope CLI
date: 2025-12-09
tags: [modelscope, cli, tools]
---

# ModelScope CLI 使用指南

ModelScope CLI 是与 ModelScope 社区（Hub）交互的命令行工具，最核心的功能是**下载模型**，其次是上传模型和登录认证。它随 `modelscope` Python 库自动安装。

---

## 1. 安装与验证

确保已安装 modelscope：

```bash
pip install modelscope --upgrade
modelscope --version
```

---

## 2. 下载模型 (`download`)

这是最常用的功能，支持断点续传，无需编写 Python 脚本即可获取模型文件。

### 下载完整模型

指定模型 ID 和本地保存路径。如果不指定路径，默认会存到系统的 Cache 目录。

```bash
# 基本用法：下载 Qwen-7B-Chat 到当前目录下的 'downloaded_models' 文件夹
modelscope download --model 'qwen/Qwen-7B-Chat' --local_dir './downloaded_models'
```

### 下载指定版本 (`--revision`)

如果不指定，默认下载 `master` 分支。对于生产环境，强烈建议指定版本号（tag）。

```bash
# 下载 v1.0.0 版本
modelscope download --model 'qwen/Qwen-7B-Chat' --revision 'v1.0.0'
```

### 只下载特定文件 (`--include` / `--exclude`)

大模型文件很大，有时你只需要 `config.json` 或者量化后的 `.gguf` 文件。

```bash
# 只下载 .json 结尾的配置文件
modelscope download --model 'qwen/Qwen-7B-Chat' --include '*.json'

# 只下载 GGUF 格式的文件（支持通配符）
modelscope download --model 'qwen/Qwen-7B-Chat-GGUF' --include '*.gguf'

# 排除 pytorch_model.bin 这种大文件
modelscope download --model 'qwen/Qwen-7B-Chat' --exclude '*.bin'
```

---

## 3. 身份认证 (`login`)

下载公开模型不需要登录。如果你需要**下载私有模型**或**上传模型**，必须先登录。

1. 去 [ModelScope 官网](https://www.modelscope.cn/) -> 个人中心 -> 访问令牌 (Access Token) 获取 Key。
2. 执行命令：

```bash
modelscope login --token <你的SDK_TOKEN>
```

*登录状态会保存在本地 `~/.modelscope/credentials` 中。*

---

## 4. 上传模型 (`upload`)

将本地模型推送到 ModelScope 社区。

```bash
# 将本地 'my_local_model_dir' 目录下的所有文件上传到 'my_username/my_model_name' 仓库
modelscope upload --model_id 'my_username/my_model_name' --files './my_local_model_dir'
```

---

## 5. 常用参数速查表

| 命令 (`modelscope <cmd>`) | 关键参数 | 说明 |
| :--- | :--- | :--- |
| **download** | `--model` | **(必选)** 模型ID (如 `damo/nlp...`) |
| | `--local_dir` | 指定下载到本地的文件夹路径 |
| | `--revision` | 指定分支名或 Tag (如 `v1.0.2`) |
| | `--include` | 只下载匹配的文件 (如 `*.json`, `*.safetensors`) |
| | `--exclude` | 排除匹配的文件 (如 `*.bin`) |
| **login** | `--token` | 使用 Access Token 进行身份验证 |
| **upload** | `--model_id` | 目标仓库 ID |
| | `--files` | 要上传的本地文件或目录 |

---

## 6. 缓存管理

默认情况下，CLI 下载的模型会存储在缓存目录，以避免重复下载。

- **Linux/Mac 默认路径:** `~/.cache/modelscope/hub/`
- **Windows 默认路径:** `C:\Users\%USERNAME%\.cache\modelscope\hub\`

如果想修改默认缓存路径，可以设置环境变量：

```bash
export MODELSCOPE_CACHE=/data/modelscope_cache
```

---

## 命令速查表

| 任务 | 命令示例 |
| :--- | :--- |
| **安装** | `pip install modelscope --upgrade` |
| **登录** | `modelscope login --token <TOKEN>` |
| **下载模型** | `modelscope download --model 'qwen/Qwen-7B-Chat'` |
| **下载到本地目录** | `modelscope download --model 'qwen/Qwen-7B-Chat' --local_dir ./models` |
| **下载指定版本** | `modelscope download --model 'qwen/Qwen-7B-Chat' --revision 'v1.0.0'` |
| **只下载配置文件** | `modelscope download --model 'qwen/Qwen-7B-Chat' --include '*.json'` |
| **上传模型** | `modelscope upload --model_id 'user/repo' --files './model_dir'` |

如需查看某个子命令的详细参数，可以使用 `--help`，例如 `modelscope download --help`。
