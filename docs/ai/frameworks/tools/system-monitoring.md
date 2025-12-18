---
title: btop 系统监控
sidebar_label: btop 监控
date: 2025-12-18
last_update:
  date: 2025-12-18
---

# btop 系统监控

btop 是一个现代化的资源监控工具，界面美观、功能强大，是 htop 的升级替代品。

## 安装

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

## 主要特性

- 实时显示 CPU、内存、磁盘、网络使用情况
- 支持鼠标操作
- 可自定义主题和布局
- 低资源占用
- 支持进程树视图

## 快捷键

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

## 配置文件

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
