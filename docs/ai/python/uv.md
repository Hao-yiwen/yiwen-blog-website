# 🚀 uv 常用命令大全

## 📦 项目管理

### 创建项目
```bash
# 创建基本项目
uv init my-project

# 指定Python版本创建
uv init --python 3.11 my-project

# 创建应用项目（包含main.py）
uv init --app my-app

# 创建库项目
uv init --lib my-lib

# 在当前目录初始化
uv init .
```

### 依赖管理
```bash
# 添加依赖
uv add requests            # 生产依赖
uv add pytest --dev       # 开发依赖
uv add "fastapi>=0.100"    # 指定版本
uv add --optional web fastapi  # 可选依赖组

# 从文件添加依赖
uv add --requirements requirements.txt

# 移除依赖
uv remove requests
uv remove pytest --dev

# 安装所有依赖
uv sync                    # 根据lock文件安装
uv sync --dev             # 包含开发依赖
uv sync --no-dev          # 只安装生产依赖
```

## 🐍 Python版本管理

```bash
# 查看可用Python版本
uv python list
uv python list --only-installed

# 安装Python版本
uv python install 3.12
uv python install 3.11.8

# 查找Python版本
uv python find 3.11

# 为项目固定Python版本
uv python pin 3.11
uv python pin cpython@3.12.1
```

## 🏃‍♂️ 运行和执行

```bash
# 运行Python脚本
uv run python main.py
uv run python -m pytest

# 运行命令
uv run --with requests python -c "import requests; print('ok')"

# 直接运行工具
uv run pytest
uv run black .
uv run mypy src/

# 临时安装并运行
uv tool run black --check .
uv tool run ruff check src/
```

## 🔧 工具管理

```bash
# 全局安装工具
uv tool install black
uv tool install ruff
uv tool install "jupyterlab>=4"

# 查看已安装工具
uv tool list

# 更新工具
uv tool upgrade black
uv tool upgrade --all

# 卸载工具
uv tool uninstall black

# 运行全局工具
uv tool run black .
```

## 📋 信息查看

```bash
# 查看项目依赖树
uv tree

# 查看过时的包
uv pip list --outdated

# 显示包信息
uv pip show requests

# 检查项目状态
uv check

# 查看lockfile
cat uv.lock
```

## 🌐 虚拟环境

```bash
# uv自动管理虚拟环境，但你也可以手动操作

# 查看虚拟环境位置
uv venv --show-path

# 创建虚拟环境
uv venv
uv venv .venv --python 3.11

# 激活虚拟环境（通常不需要，uv run会自动处理）
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

## 🔄 pip兼容命令

```bash
# uv也支持pip风格的命令
uv pip install requests
uv pip install -r requirements.txt
uv pip uninstall requests
uv pip list
uv pip freeze
uv pip show requests
```

## 📚 实际工作流示例

### 新项目完整流程
```bash
# 1. 创建项目
uv init --python 3.11 my-web-app
cd my-web-app

# 2. 添加依赖
uv add fastapi uvicorn
uv add pytest black --dev

# 4. 运行应用
uv run uvicorn main:app --reload

# 5. 运行测试
uv run pytest

# 6. 代码格式化
uv run black .
```

### 日常开发命令
```bash
# 启动开发服务器
uv run python manage.py runserver

# 运行测试套件
uv run pytest tests/ -v

# 类型检查
uv run mypy src/

# 代码质量检查
uv run ruff check .
uv run black --check .

# 安装新依赖
uv add pandas numpy
```

## 🎯 高级用法

### 工作空间管理
```bash
# 多包项目
uv add --editable ./packages/core
uv sync --package my-package
```

### 脚本模式
```bash
# 创建单文件脚本
uv init --script my-script.py

# 运行脚本（自动安装依赖）
uv run my-script.py
```

### 缓存管理
```bash
# 清理缓存
uv cache clean

# 查看缓存大小
uv cache dir
```

## 🔍 配置和调试

```bash
# 查看uv版本
uv --version

# 查看详细信息
uv --verbose run python main.py

# 查看帮助
uv --help
uv add --help
```

## 💡 最佳实践组合

**日常最常用的命令：**
```bash
uv init --python 3.11 project-name  # 创建项目
uv add package-name                  # 添加依赖  
uv run python main.py               # 运行代码
uv sync                             # 同步依赖
```

**开发调试常用：**
```bash
uv run pytest                      # 测试
uv run black .                     # 格式化
uv tree                           # 查看依赖
uv python list                    # 管理Python版本
```