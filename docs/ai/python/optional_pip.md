---
title: Python pip 中括号语法详解
sidebar_label: "pip 可选依赖 [extras]"
---

## 中括号的含义

```bash
pip install fastapi[standard]
#               包名 [可选依赖组名]
```

这表示：安装 `fastapi` **并且**安装名为 `standard` 的可选依赖组。

## 工作原理

### 在 `pyproject.toml` 中定义

FastAPI 的配置文件大概是这样的：

```toml
[project]
name = "fastapi"
dependencies = [
    "starlette>=0.27.0",
    "pydantic>=1.7.4",
    "typing-extensions>=4.8.0"
]

[project.optional-dependencies]
# 基础依赖 - 只有核心功能
standard = [
    "uvicorn[standard]>=0.12.0",
    "pydantic-settings>=2.0.0",
    "python-multipart>=0.0.5",
    "email-validator>=2.0.0",
    "httpx>=0.23.0",
    "jinja2>=2.11.2",
]

# 所有依赖
all = [
    "uvicorn[standard]",
    "pydantic-settings",
    "python-multipart",
    "email-validator",
    "httpx",
    "jinja2",
    "itsdangerous",
    "pyyaml",
    "ujson",
    "orjson",
]

# 开发依赖
dev = [
    "pytest>=7.1.3",
    "mypy==1.4.1",
    "ruff==0.0.292",
]
```

## 安装效果对比

```bash
# 1. 只装核心（最小依赖）
pip install fastapi
# 只装：fastapi + starlette + pydantic

# 2. 装标准版（推荐）
pip install fastapi[standard]
# 装：fastapi + starlette + pydantic + uvicorn + 其他常用工具

# 3. 装完整版
pip install fastapi[all]
# 装：所有可选功能

# 4. 装多个 extras
pip install fastapi[standard,dev]
# 装：标准版 + 开发工具
```

## 常见的 extras 例子

### FastAPI
```bash
fastapi[standard]  # 生产环境推荐
fastapi[all]       # 所有功能
```

### Uvicorn
```bash
uvicorn            # 基础版，性能一般
uvicorn[standard]  # 标准版，包含性能优化
#   ↓ 包含：
#   - uvloop (更快的事件循环)
#   - httptools (更快的 HTTP 解析)
#   - websockets (WebSocket 支持)
```

### SQLAlchemy
```bash
sqlalchemy                    # 核心
sqlalchemy[asyncio]          # 异步支持
sqlalchemy[postgresql]       # PostgreSQL 驱动
sqlalchemy[mysql]            # MySQL 驱动
sqlalchemy[asyncio,postgresql]  # 多个 extras
```

### Requests
```bash
requests              # 基础 HTTP 库
requests[security]    # 添加安全功能
requests[socks]       # SOCKS 代理支持
```

### Pandas
```bash
pandas                # 核心功能
pandas[performance]   # 性能优化（numexpr, bottleneck）
pandas[plot]          # 绘图功能（matplotlib）
pandas[excel]         # Excel 支持（openpyxl）
pandas[all]           # 所有功能
```

## 实际例子

```bash
# 创建一个完整的 FastAPI 项目
uv add "fastapi[standard]"

# 相当于手动安装：
uv add fastapi
uv add "uvicorn[standard]"
uv add pydantic-settings
uv add python-multipart
uv add email-validator
uv add httpx
uv add jinja2
```

## 为什么要这样设计？

### 1. **按需安装**
```python
# 场景 1：只需要 FastAPI 框架（如用于类型检查）
pip install fastapi  # 轻量级，几秒装完

# 场景 2：生产环境
pip install fastapi[standard]  # 包含所有常用工具

# 场景 3：开发环境
pip install fastapi[standard,dev]  # 额外包含测试工具
```

### 2. **避免依赖冲突**
```bash
# 如果你已经有特定版本的 uvicorn
pip install uvicorn==0.20.0
pip install fastapi  # 不会覆盖你的 uvicorn

# 但如果你用 [standard]
pip install fastapi[standard]  # 可能升级你的 uvicorn
```

### 3. **减少包体积**
```
fastapi 核心：       ~500 KB
fastapi[standard]：  ~5 MB
fastapi[all]：       ~20 MB
```

## 查看一个包有哪些 extras

```bash
# 方法 1：查看 PyPI 页面
https://pypi.org/project/fastapi/

# 方法 2：查看包的 pyproject.toml
pip download --no-deps fastapi
tar -xzf fastapi-*.tar.gz
cat fastapi-*/pyproject.toml

# 方法 3：用 pip show（不太详细）
pip show fastapi
```

## 在你自己的项目中定义 extras

```toml
# pyproject.toml
[project]
name = "my-awesome-app"
dependencies = [
    "fastapi",
]

[project.optional-dependencies]
# 开发环境
dev = [
    "pytest",
    "black",
    "mypy",
]

# 生产环境
prod = [
    "uvicorn[standard]",
    "gunicorn",
]

# 数据库
postgres = [
    "psycopg2-binary",
    "sqlalchemy[asyncio]",
]

mysql = [
    "pymysql",
    "sqlalchemy[asyncio]",
]
```

然后安装：
```bash
# 开发
pip install -e ".[dev]"

# 生产 + PostgreSQL
pip install ".[prod,postgres]"
```

## 总结

**中括号 `[]` = 可选依赖组**

```bash
包名[extras名称]
  ↓     ↓
fastapi[standard]
```

**作用**：
- ✅ 按需安装，不浪费空间
- ✅ 避免依赖冲突
- ✅ 清晰标记功能模块

**最佳实践**：
```bash
# 日常开发
uv add "fastapi[standard]"

# 如果不确定需要什么
uv add fastapi  # 先装最小版本
# 需要什么功能再加
```