# Hugging Face 模型上传完整教程（Git方式）

## 🔐 第一步：登录认证

```bash
# 1. 安装 huggingface-cli（如果还没安装）
pip install huggingface_hub

# 2. 登录 Hugging Face
huggingface-cli login

# 会提示：
# Token: 
# 输入你的 token（从 https://huggingface.co/settings/tokens 获取）
# 
# Add token as git credential? (Y/n) 
# 输入 Y （这样 git push 时就不用再输密码了）
```

### 获取 Token 步骤：
1. 访问：https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择 "Write" 权限
4. 复制生成的 token（hf_xxxxx）

---

## 🏗️ 第二步：创建远程仓库

```bash
# 在 HF 上创建新仓库
huggingface-cli repo create MiniMind-New --type model

# 输出示例：
# https://huggingface.co/yiwenX/MiniMind-New
```

---

## 📦 第三步：本地初始化 Git 仓库

```bash
# 1. 进入你的模型文件夹
cd /root/minimind/MiniMind2

# 2. 初始化 git 仓库
git init

# 3. 安装并配置 Git LFS
git lfs install

# 4. 配置 LFS 追踪大文件（非常重要！）
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.msgpack"
git lfs track "*.model"
git lfs track "*.h5"

# 5. 添加 .gitattributes 到 git
git add .gitattributes

# 6. 查看 LFS 配置（确认是否正确）
cat .gitattributes
```

---

## 🔗 第四步：关联远程仓库

```bash
# 添加 HF 远程仓库（注意替换用户名和仓库名）
git remote add origin https://huggingface.co/yiwenX/MiniMind-New

# 验证远程仓库
git remote -v
# 应该看到：
# origin  https://huggingface.co/yiwenX/MiniMind-New (fetch)
# origin  https://huggingface.co/yiwenX/MiniMind-New (push)
```

---

## 📤 第五步：提交并推送

```bash
# 1. 添加所有文件
git add .

# 2. 查看将要提交的文件
git status

# 3. 提交到本地仓库
git commit -m "Initial commit: Upload MiniMind model"

# 4. 设置主分支为 main
git branch -M main

# 5. 推送到 HF
git push -u origin main

# 注意：首次推送大文件会比较慢，耐心等待
# 你会看到 LFS 上传进度
```

## ⚠️ 常见问题

### 1. Git LFS 未安装

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# CentOS/RHEL
sudo yum install git-lfs

# 安装后初始化
git lfs install
```

### 2. 推送失败：认证问题

```bash
# 重新登录
huggingface-cli login

# 或手动配置 git credential
git config --global credential.helper store
# 然后 push 时输入用户名和 token
```

### 3. 大文件上传超时

```bash
# 增加 git 超时时间
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

# 或分批上传大文件
git add model.pth
git commit -m "Add model weights"
git push
```

### 4. 查看 LFS 文件

```bash
# 查看被 LFS 追踪的文件
git lfs ls-files

# 查看 LFS 存储使用情况
git lfs status
```