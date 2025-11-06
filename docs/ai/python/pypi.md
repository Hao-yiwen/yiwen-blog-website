---
title: "如何发布 PyPI 包 - 完整教程"
sidebar_label: "如何发布 PyPI 包 - 完整教程"
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# 如何发布 PyPI 包 - 完整教程

## 目录
1. [准备工作](#准备工作)
2. [项目结构](#项目结构)
3. [配置文件](#配置文件)
4. [打包](#打包)
5. [上传到 PyPI](#上传到-pypi)
6. [完整示例](#完整示例)

---

## 准备工作

### 1. 安装必要工具

```bash
# 安装打包工具
pip install build twine

# build: 用于构建包
# twine: 用于上传到 PyPI
```

### 2. 注册 PyPI 账号

- **PyPI**（正式）: https://pypi.org/account/register/

### 3. 生成 API Token（推荐）

1. 登录 PyPI
2. 进入 Account settings → API tokens
3. 点击 "Add API token"
4. 保存 token（只显示一次！）

---

## 项目结构

### 标准结构

```
my_awesome_package/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── module1.py
│       └── module2.py
├── tests/
│   ├── __init__.py
│   └── test_module1.py
├── README.md
├── LICENSE
├── pyproject.toml  ← 核心配置文件
└── setup.py        ← 可选（传统方式）
```

### 简单结构（小项目）

```
my_simple_package/
├── my_package/
│   ├── __init__.py
│   └── core.py
├── README.md
├── LICENSE
└── pyproject.toml
```

---

## 配置文件

### 现代方式：pyproject.toml（推荐）

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-awesome-package"  # PyPI 上的包名（用连字符）
version = "0.1.0"
description = "一个很棒的 Python 包"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
requires-python = ">=3.7"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
keywords = ["example", "tutorial", "package"]

# 依赖项
dependencies = [
    "numpy>=1.20.0",
    "requests>=2.25.0",
]

# 可选依赖
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=22.0",
    "flake8>=4.0",
]
docs = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
]

# 项目 URL
[project.urls]
Homepage = "https://github.com/yourusername/my-awesome-package"
Documentation = "https://my-awesome-package.readthedocs.io"
Repository = "https://github.com/yourusername/my-awesome-package"
"Bug Tracker" = "https://github.com/yourusername/my-awesome-package/issues"

# 命令行工具（可选）
[project.scripts]
my-cli = "my_package.cli:main"

# 如果使用 src 布局
[tool.setuptools.packages.find]
where = ["src"]

# 如果直接在根目录
# [tool.setuptools.packages.find]
# where = ["."]
# include = ["my_package*"]
```

### 传统方式：setup.py（可选）

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="my-awesome-package",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="一个很棒的 Python 包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my-awesome-package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0"],
    },
)
```

### README.md

```markdown
# My Awesome Package

一个很棒的 Python 包。

## 安装

```bash
pip install my-awesome-package
```

## 快速开始

```python
from my_package import hello

hello.greet("World")
# 输出: Hello, World!
```

## 功能特性

- 特性 1
- 特性 2
- 特性 3

## 文档

详细文档请访问：https://my-awesome-package.readthedocs.io

## 许可证

MIT License
```

### LICENSE

```text
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 打包

### 1. 清理旧文件

```bash
# 删除旧的构建文件
rm -rf build/ dist/ *.egg-info
```

### 2. 构建包

```bash
# 使用 build 工具（推荐）
python -m build

# 或者使用 setup.py（传统）
python setup.py sdist bdist_wheel
```

这会在 `dist/` 目录生成两个文件：
- `my_awesome_package-0.1.0.tar.gz` (源码分发)
- `my_awesome_package-0.1.0-py3-none-any.whl` (wheel 分发)

### 3. 检查包

```bash
# 检查包的元数据
twine check dist/*
```

---

## 上传到 PyPI

### 方法 1：使用 API Token（推荐）

#### 配置 Token

创建 `~/.pypirc` 文件：

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmcC...  # 你的 API token

[testpypi]
  username = __token__
  password = pypi-AgENdGVzdC5weXBp...  # TestPyPI 的 token
```

#### 上传到 TestPyPI（测试）

```bash
# 先在测试环境试试
twine upload --repository testpypi dist/*

# 安装测试
pip install --index-url https://test.pypi.org/simple/ my-awesome-package
```

#### 上传到 PyPI（正式）

```bash
# 确认无误后上传到正式 PyPI
twine upload dist/*
```

### 方法 2：手动输入密码

```bash
# 会提示输入用户名和密码
twine upload dist/*
```

### 方法 3：环境变量

```bash
# 设置环境变量
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmcC...

# 上传
twine upload dist/*
```

---

## 完整示例

让我们创建一个完整的示例包：

### 1. 创建项目结构

```bash
mkdir hello-pypi && cd hello-pypi

# 创建包目录
mkdir -p hello_pypi
touch hello_pypi/__init__.py
touch hello_pypi/greetings.py

# 创建配置文件
touch pyproject.toml
touch README.md
touch LICENSE
```

### 2. 编写代码

**hello_pypi/__init__.py**
```python
"""一个简单的问候包"""

__version__ = "0.1.0"

from .greetings import hello, goodbye

__all__ = ["hello", "goodbye"]
```

**hello_pypi/greetings.py**
```python
def hello(name: str) -> str:
    """向某人问好"""
    return f"Hello, {name}!"

def goodbye(name: str) -> str:
    """向某人道别"""
    return f"Goodbye, {name}!"
```

### 3. 配置 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "hello-pypi-example"
version = "0.1.0"
description = "一个简单的问候包示例"
readme = "README.md"
authors = [{name = "Your Name", email = "your@email.com"}]
license = {text = "MIT"}
requires-python = ">=3.7"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]

[project.urls]
Homepage = "https://github.com/yourusername/hello-pypi"
```

### 4. 编写 README.md

```markdown
# Hello PyPI Example

一个简单的 Python 包示例。

## 安装

```bash
pip install hello-pypi-example
```

## 使用

```python
from hello_pypi import hello, goodbye

print(hello("World"))     # Hello, World!
print(goodbye("Python"))  # Goodbye, Python!
```
```

### 5. 打包并上传

```bash
# 1. 构建
python -m build

# 2. 检查
twine check dist/*

# 3. 测试上传
twine upload --repository testpypi dist/*

# 4. 测试安装
pip install --index-url https://test.pypi.org/simple/ hello-pypi-example

# 5. 测试使用
python -c "from hello_pypi import hello; print(hello('PyPI'))"

# 6. 正式上传
twine upload dist/*
```

---

## 版本管理

### 更新版本

修改 `pyproject.toml` 中的版本号：

```toml
[project]
version = "0.2.0"  # 从 0.1.0 更新到 0.2.0
```

### 语义化版本

遵循 SemVer 规范：`MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的 API 修改
- **MINOR**: 向下兼容的功能新增
- **PATCH**: 向下兼容的问题修正

示例：
- `0.1.0` → `0.1.1` (bug 修复)
- `0.1.1` → `0.2.0` (新功能)
- `0.2.0` → `1.0.0` (重大更新)

---

## 常见问题

### 1. 包名已被占用

```bash
# 错误：The name 'xxx' is already taken
```

**解决**: 换一个唯一的包名，可以加前缀或后缀

### 2. 上传失败：403 错误

```bash
# 403 Forbidden
```

**原因**: 
- Token 错误或过期
- 没有权限

**解决**: 重新生成 token

### 3. 版本冲突

```bash
# File already exists
```

**原因**: 该版本已上传过

**解决**: 更新版本号，PyPI 不允许覆盖已有版本

### 4. 导入问题

```python
# ModuleNotFoundError: No module named 'my_package'
```

**检查**:
- `pyproject.toml` 中的包配置是否正确
- 是否有 `__init__.py`
- 包名是否正确（导入用下划线，PyPI 名可用连字符）

---

## 最佳实践

### ✅ DO

1. **使用有意义的包名**: 清晰、简短、易记
2. **写好 README**: 包含安装、使用、示例
3. **添加测试**: 使用 pytest 等测试框架
4. **语义化版本**: 遵循 SemVer
5. **添加 LICENSE**: 明确开源协议
6. **写文档**: 使用 Sphinx 或 MkDocs
7. **先测试**: 先上传到 TestPyPI
8. **使用 API Token**: 比密码更安全
9. **添加类型提示**: 使用 Type Hints
10. **持续集成**: 使用 GitHub Actions

### ❌ DON'T

1. ❌ 使用太通用的包名
2. ❌ 上传没测试过的代码
3. ❌ 忘记更新版本号
4. ❌ 包含敏感信息（密码、token）
5. ❌ 忽略依赖版本管理

---

## 进阶话题

### 1. 添加命令行工具

```toml
[project.scripts]
my-tool = "my_package.cli:main"
```

```python
# my_package/cli.py
def main():
    print("Hello from CLI!")

if __name__ == "__main__":
    main()
```

安装后可以直接运行：
```bash
my-tool
```

### 2. 包含数据文件

```toml
[tool.setuptools.package-data]
my_package = ["data/*.json", "templates/*.html"]
```

### 3. 使用 GitHub Actions 自动发布

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        pip install build twine
    - name: Build and publish
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        python -m build
        twine upload dist/*
```

---

## 快速检查清单

发布前检查：

- [ ] 代码功能正常
- [ ] 有单元测试
- [ ] README.md 完整
- [ ] LICENSE 文件存在
- [ ] 版本号已更新
- [ ] pyproject.toml 配置正确
- [ ] 在 TestPyPI 测试成功
- [ ] 所有依赖已列出
- [ ] 文档链接有效
- [ ] 没有敏感信息

---

## 有用的链接

- **PyPI**: https://pypi.org
- **TestPyPI**: https://test.pypi.org
- **Python 打包指南**: https://packaging.python.org
- **setuptools 文档**: https://setuptools.pypa.io
- **Twine 文档**: https://twine.readthedocs.io

---

**恭喜！现在你可以发布自己的 PyPI 包了！** 🎉