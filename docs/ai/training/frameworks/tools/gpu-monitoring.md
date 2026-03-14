---
title: GPU 状态监控
sidebar_position: 15
tags: [nvidia, gpu, monitoring, nvidia-smi, nvtop]
---

# GPU 状态监控

在深度学习训练和推理过程中，持续监控 GPU 状态是非常重要的。以下是几种常用的监控方式。

## nvidia-smi 自带的循环参数

最简单的方式是使用 `nvidia-smi` 自带的循环刷新功能：

```bash
nvidia-smi -l 1      # 每 1 秒刷新一次
nvidia-smi -lms 500  # 每 500 毫秒刷新一次
```

## watch 命令（推荐）

使用 `watch` 命令可以获得更清晰的显示效果：

```bash
watch -n 1 nvidia-smi   # 每 1 秒刷新
watch -n 0.5 nvidia-smi # 每 0.5 秒刷新
```

### 只查看关键信息

如果只想看核心指标，可以使用查询参数：

```bash
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv'
```

输出示例：
```
utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], temperature.gpu
85 %, 45 %, 18432 MiB, 24576 MiB, 72
```

## nvtop（交互式工具）

[nvtop](https://github.com/Syllo/nvtop) 是一个类似 `htop` 的交互式 GPU 监控工具，提供实时曲线图和进程信息。

### 安装

```bash
# Ubuntu/Debian
sudo apt install nvtop

# CentOS/RHEL
sudo yum install nvtop

# macOS (仅支持 Apple Silicon)
brew install nvtop
```

### 使用

```bash
nvtop
```

nvtop 的优势：
- 实时 GPU 使用率曲线图
- 显存使用情况可视化
- 进程级别的 GPU 占用信息
- 支持多 GPU 监控
- 交互式操作（排序、筛选等）

## gpustat（简洁美观）

[gpustat](https://github.com/wookayin/gpustat) 是一个轻量级的 GPU 状态查看工具，输出简洁美观。

### 安装

```bash
pip install gpustat
```

### 使用

```bash
gpustat           # 单次查看
gpustat -i 1      # 每秒刷新
gpustat -cp       # 显示进程信息和完整命令
gpustat --watch   # 持续监控模式
```

## 工具对比

| 工具 | 安装方式 | 特点 | 推荐场景 |
|------|----------|------|----------|
| `nvidia-smi -l` | 预装 | 无需安装，信息全面 | 快速查看 |
| `watch nvidia-smi` | 预装 | 显示清晰，高亮变化 | 临时监控 |
| `nvtop` | apt/yum | 交互式，可视化曲线 | 长期监控 |
| `gpustat` | pip | 简洁美观，支持远程 | 多卡概览 |

## 常用监控指标说明

- **GPU Utilization**: GPU 核心利用率，理想状态应接近 100%
- **Memory Used/Total**: 显存使用量/总量
- **Temperature**: GPU 温度，通常应保持在 80°C 以下
- **Power Draw**: 功耗，可用于判断是否达到功耗墙
- **SM Clock**: GPU 核心频率

## 小技巧

### 远程监控

结合 `watch` 和 SSH 可以远程监控：

```bash
watch -n 1 'ssh user@server nvidia-smi'
```

### 记录 GPU 状态日志

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 5 > gpu_log.csv
```

这会每 5 秒记录一次 GPU 状态到 CSV 文件，方便后续分析。
