---
title: 系统监控工具
sidebar_label: 系统监控
date: 2025-12-18
last_update:
  date: 2025-12-18
---

# 系统监控工具

在进行深度学习训练时，监控系统资源（CPU、内存、GPU、网络等）非常重要。本文介绍几种常用的系统监控工具。

## btop

btop 是一个现代化的资源监控工具，界面美观、功能强大，是 htop 的升级替代品。

### 安装

```bash
# Ubuntu/Debian
sudo apt install btop

# CentOS/RHEL (需要 EPEL)
sudo dnf install epel-release
sudo dnf install btop

# macOS
brew install btop

# Arch Linux
sudo pacman -S btop
```

### 主要特性

- 实时显示 CPU、内存、磁盘、网络使用情况
- 支持鼠标操作
- 可自定义主题和布局
- 低资源占用
- 支持进程树视图

### 界面说明

```
┌─ CPU ──────────────────────────────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 25%  │
│ Core 0: 30%  Core 1: 20%  Core 2: 25%  ...    │
└────────────────────────────────────────────────┘
┌─ Memory ───────────────────────────────────────┐
│ RAM: ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░ 8.5G / 16G       │
│ Swap: ▓░░░░░░░░░░░░░░░░░░░░░ 512M / 8G        │
└────────────────────────────────────────────────┘
```

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `h` | 显示帮助 |
| `m` | 切换内存显示模式 |
| `p` | 切换进程排序方式 |
| `t` | 切换进程树视图 |
| `f` | 过滤进程 |
| `k` | 杀死选中进程 |
| `q` | 退出 |
| `Esc` | 关闭菜单/取消 |

### 配置文件

配置文件位于 `~/.config/btop/btop.conf`，可自定义：

```bash
# 主题设置
color_theme = "Default"

# 刷新率 (毫秒)
update_ms = 2000

# 显示进程数
proc_per_core = false

# 显示 CPU 温度
show_cpu_freq = true
```

## nvidia-smi

用于监控 NVIDIA GPU 的官方工具。

### 常用命令

```bash
# 查看 GPU 状态
nvidia-smi

# 持续监控 (每秒刷新)
nvidia-smi -l 1

# 简洁模式持续监控
watch -n 1 nvidia-smi

# 查看进程详情
nvidia-smi pmon -i 0

# 查询特定信息
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

### 输出解读

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM    On    | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    52W / 400W |   1024MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

- **Memory-Usage**: 显存使用量
- **GPU-Util**: GPU 利用率
- **Pwr:Usage/Cap**: 功耗/最大功耗
- **Temp**: GPU 温度

## nvitop

一个交互式 NVIDIA GPU 进程查看器，比 nvidia-smi 更友好。

### 安装

```bash
pip install nvitop
```

### 使用

```bash
# 启动监控
nvitop

# 监控模式
nvitop -m
```

### 特性

- 类似 htop 的交互界面
- 显示每个进程的 GPU 内存占用
- 支持进程过滤和排序
- 内存和利用率的历史图表

## gpustat

轻量级 GPU 状态查看工具。

### 安装与使用

```bash
pip install gpustat

# 查看状态
gpustat

# 持续监控
gpustat -i 1

# 显示完整命令
gpustat -p
```

## htop

经典的系统监控工具。

### 安装

```bash
sudo apt install htop
```

### 常用快捷键

| 快捷键 | 功能 |
|--------|------|
| `F2` | 设置 |
| `F3` | 搜索进程 |
| `F4` | 过滤进程 |
| `F5` | 树形视图 |
| `F6` | 排序 |
| `F9` | 杀死进程 |
| `F10` | 退出 |

## 监控最佳实践

### 训练时监控脚本

创建一个监控脚本，同时查看 CPU 和 GPU：

```bash
#!/bin/bash
# monitor.sh

# 分屏显示 btop 和 nvidia-smi
tmux new-session -d -s monitor
tmux split-window -h
tmux send-keys -t 0 'btop' C-m
tmux send-keys -t 1 'watch -n 1 nvidia-smi' C-m
tmux attach-session -t monitor
```

### 日志记录

将 GPU 状态记录到文件：

```bash
# 每 10 秒记录一次 GPU 状态
while true; do
    echo "=== $(date) ===" >> gpu_log.txt
    nvidia-smi >> gpu_log.txt
    sleep 10
done
```

### 远程监控

使用 SSH 远程查看服务器状态：

```bash
# 直接查看
ssh user@server "nvidia-smi"

# 持续监控
ssh -t user@server "watch -n 1 nvidia-smi"
```

## 工具对比

| 工具 | 用途 | 特点 |
|------|------|------|
| btop | 系统资源 | 界面美观，功能全面 |
| htop | 系统资源 | 经典可靠 |
| nvidia-smi | GPU | 官方工具，信息最全 |
| nvitop | GPU | 交互式，更友好 |
| gpustat | GPU | 轻量简洁 |

:::tip 推荐组合
日常监控：btop + nvitop

训练时：使用 tmux 分屏同时运行 btop 和 nvidia-smi
:::
