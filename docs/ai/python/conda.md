# 🐍 Conda 常用命令（简化版）

## 安装 Conda
```bash
# macOS (推荐 Miniforge，更快更轻量)
brew install miniforge

# 或下载 Miniconda
# https://docs.conda.io/en/latest/miniconda.html
```

## 📦 基本工作流

### 1. 创建环境
```bash
conda create -n myenv python=3.11          # 创建环境
conda create -n myenv python=3.11 numpy    # 同时安装包
```

### 2. 激活/切换环境
```bash
conda activate myenv        # 激活环境
conda deactivate           # 退出环境
```

### 3. 管理依赖
```bash
conda install pandas       # 安装包
conda install numpy=1.24   # 指定版本
conda remove pandas        # 删除包
conda update pandas        # 更新包
```

### 4. 运行代码
```bash
# 激活环境后直接运行
conda activate myenv
python main.py
```

## 🌍 环境管理
```bash
conda env list             # 查看所有环境
conda env remove -n myenv  # 删除环境
```

## 📋 查看信息
```bash
conda list                 # 当前环境的包
conda list -n myenv        # 指定环境的包
conda search numpy         # 搜索可用包版本
```

## 📤 导入/导出环境
```bash
# 导出环境
conda env export > environment.yml

# 从文件创建环境
conda env create -f environment.yml
```

## 💡 实际例子

### 创建数据科学环境
```bash
conda create -n ds python=3.11
conda activate ds
conda install pandas numpy matplotlib scikit-learn jupyter
jupyter lab
```

### 创建深度学习环境
```bash
conda create -n ml python=3.11
conda activate ml
conda install pytorch torchvision -c pytorch
python train.py
```

## ⚙️ 常用配置

```bash
# 禁用base自动激活
conda config --set auto_activate_base false

# 添加conda-forge频道
conda config --add channels conda-forge

# 清理缓存
conda clean --all
```

## 🎯 记住这6个命令就够了
```bash
conda create -n name python=3.11  # 创建环境
conda activate name               # 激活环境
conda install package            # 安装包
conda list                       # 查看包
conda env list                   # 查看环境
conda deactivate                 # 退出环境
```