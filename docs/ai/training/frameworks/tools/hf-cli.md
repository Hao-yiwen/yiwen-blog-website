---
title: Hugging Face CLI (hf) 使用指南
sidebar_label: HF CLI
date: 2025-12-06
tags: [huggingface, cli, tools]
---

# Hugging Face CLI (`hf`) 使用指南

## 1. 安装与验证

要使用 `hf` 命令，下面是主要方式安装：

### 安装独立二进制文件 (推荐，无需 Python 环境)

这会安装一个独立的 `hf` 可执行文件，速度更快且不依赖本地 Python 环境。

* **Linux/macOS:**
  ```bash
  curl -LsSf https://hf.co/cli/install.sh | bash
  ```

**验证安装：**

```bash
hf --version
```

---

## 2. 认证 (Authentication)

在下载私有模型或上传文件前，必须先登录。

* **登录：** (交互式，需要 Access Token)
  ```bash
  hf auth login
  ```
* **查看当前登录状态：**
  ```bash
  hf auth whoami
  ```
* **切换账号（如果你有多个 Token）：**
  ```bash
  hf auth switch
  ```
* **登出：**
  ```bash
  hf auth logout
  ```

---

## 3. 下载 (Download)

这是 `hf` 最常用的功能。它支持断点续传，且结构比旧版更清晰。

### 下载整个模型/仓库

默认下载到缓存目录 (`~/.cache/huggingface/hub`)。

```bash
hf download gpt2
```

### 下载特定文件

只下载仓库中的 `config.json`：

```bash
hf download gpt2 config.json
```

### 下载到指定目录 (不使用缓存结构)

如果你想把文件直接下载到当前文件夹的 `./models` 目录，而不是系统缓存：

```bash
hf download gpt2 --local-dir ./models/gpt2
```

### 下载数据集 (Dataset) 或 Space

默认是下载 `model`，如果是数据集需要指定 `--repo-type`：

```bash
# 下载数据集
hf download wikipedia --repo-type dataset

# 下载 Space 代码
hf download cjeon/korean-lpr --repo-type space
```

### 包含/排除特定文件 (Glob pattern)

只下载 safetensors 文件：

```bash
hf download stabilityai/stable-diffusion-3-medium --include "*.safetensors"
```

---

## 4. 上传 (Upload)

将本地文件推送到 Hugging Face Hub。

### 上传单个文件

格式：`hf upload [REPO_ID] [本地文件] [远程路径]`

```bash
# 将本地的 my_model.bin 上传到 username/my-repo 的根目录
hf upload username/my-repo ./my_model.bin
```

### 上传整个文件夹

```bash
# 将 ./checkpoint 文件夹的内容上传到远程仓库
hf upload username/my-repo ./checkpoint
```

### 上传到数据集

```bash
hf upload username/my-dataset ./data.csv --repo-type dataset
```

---

## 5. 仓库管理 (Repository Management)

直接在终端创建或删除仓库，无需去网页端。

* **创建新仓库：**

  ```bash
  hf repo create my-new-model
  ```

  *选项：*

  * `--type dataset`：创建数据集。
  * `--organization my-org`：在组织下创建。
  * `--private`：创建私有仓库。

* **删除仓库：**

  ```bash
  hf repo delete username/my-new-model
  ```

---

## 6. 缓存管理 (Cache Management)

这是 `hf` 相比旧版的一大改进，用于清理磁盘空间。

* **交互式扫描与清理 (强烈推荐)：**
  这个命令会列出所有下载的模型及其占用的空间，允许你用方向键选择并删除旧版本。
  ```bash
  hf scan-cache
  ```

---

## 7. 高级技巧：开启超速下载

`hf` 兼容 `hf_transfer`（基于 Rust 的高速传输库），能最大化利用带宽。

1. **安装库：**
   ```bash
   pip install hf_transfer
   ```
2. **设置环境变量并运行：**
   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 hf download gpt2
   ```

---

## 命令速查表

| 任务 | 命令示例 |
| :--- | :--- |
| **登录** | `hf auth login` |
| **下载模型** | `hf download meta-llama/Llama-2-7b` |
| **下载文件** | `hf download user/repo model.safetensors` |
| **下载到本地目录** | `hf download user/repo --local-dir ./data` |
| **上传文件夹** | `hf upload user/repo ./local_folder` |
| **清理磁盘** | `hf scan-cache` |
| **查看环境信息** | `hf env` |

如果你需要关于某个具体子命令的详细参数，可以使用 `--help`，例如 `hf download --help`。
