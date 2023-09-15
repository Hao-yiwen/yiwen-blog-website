# oh-my-zsh代码提示和补全

## 安装oh-my-zsh

1. 安装

```bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

2. 编辑~/.zshrc

```bash
vim ~/.zshrc  # 或使用你喜欢的其他文本编辑器
```

3. 主题配置

```bash
ZSH_THEME="robbyrussell"  # 默认主题
```

4. 保存配置

```bash
source ~/.zshrc
```

## 使用 zsh-autosuggestions

1. 下载

```bash
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

2. 打开~/.zshrc

```bash
plugins=(git zsh-autosuggestions)
```

3. 保存

```bash
source ~/.zshrc
```
