---
title: AutoDL预设
sidebar_label: AutoDL预设
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# AutoDL预设

## 私有镜像
- 镜像源 （实测中科大镜像源速度最快）
    - 中科大镜像源: https://pypi.mirrors.ustc.edu.cn/simple
    - 清华镜像源: https://pypi.tuna.tsinghua.edu.cn/simple
```bash
# 设置中科大镜像源
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

# 如果需要添加信任主机
pip config set global.trusted-host mirrors.ustc.edu.cn
```
- clash脚本
- node 和 claude code
- vscode插件 (python/python debugger/jupter)
- 预设3个版本
    - cuda 13.0/torch 2.8.0 ✅
    - cuda 12.4/torch 2.6.0 ✅
    - cuda 11.8/torch 2.0.0 ✅

# HF预设
```bash
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1    # 打开多连接/并行管线
# 可选：限制并行数，默认已挺猛
export HF_HUB_NUM_THREADS=16
export HF_ENDPOINT='https://hf-mirror.com'
huggingface-cli download moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --include "model*.safetensors" "config.json" "tokenizer*" \
  --local-dir ./kimi-48b --local-dir-use-symlinks False \
  --max-workers 16 --resume-download
```