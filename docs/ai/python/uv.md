# 🚀 uv 常用命令（简化版）

## 安装 uv
```bash
# macOS
brew install uv

# 或通过官方脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 📦 基本工作流

### 1. 创建项目
```bash
uv init my-project          # 创建新项目（使用系统默认Python）
uv init --python 3.11 my-project  # 指定Python版本（推荐）
cd my-project
```

### 2. 管理依赖
```bash
uv add requests             # 添加包
uv add pytest --dev         # 添加开发依赖
uv remove requests          # 删除包
uv sync                     # 安装所有依赖
```

### 3. 运行代码
```bash
uv run python main.py       # 运行Python脚本
uv run pytest              # 运行测试
uv run python              # 进入Python REPL
```

## 🐍 Python版本
```bash
uv python find             # 查看默认Python版本
uv python list             # 查看可用版本
uv python install 3.12     # 安装Python版本
uv python pin 3.11         # 固定项目Python版本
```

## 🔧 全局工具
```bash
uv tool install black      # 全局安装工具
uv tool run black .        # 运行工具
uv tool list              # 查看已安装工具
```

## 📋 查看信息
```bash
uv tree                   # 查看依赖树
uv pip list              # 列出已安装包
```

## 💡 实际例子

### 创建 FastAPI 项目
```bash
uv init --python 3.11 my-api  # 明确指定版本
cd my-api
uv add fastapi uvicorn
uv run uvicorn main:app --reload
```

### 创建数据分析项目
```bash
uv init --python 3.12 data-project
cd data-project
uv add pandas numpy matplotlib
uv run python analysis.py
```

### 从 requirements.txt 迁移
```bash
uv init --python 3.11 .
uv add --requirements requirements.txt
uv sync
```

## 🎯 记住这5个命令就够了
```bash
uv init --python 3.11   # 创建项目（建议指定版本）
uv add                  # 添加依赖
uv run                  # 运行代码
uv sync                 # 同步依赖
uv python              # 管理Python版本
```