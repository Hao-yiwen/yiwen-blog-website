# 使用rbenv管理ruby

## 背景

使用rvm管理ruby在安装依赖的时候存在各种报错，然后解决很麻烦，然后看到很多人推荐rbenv，于是使用rbenv来管理ruby。

## 删除rvm使用rbenv

[文档](https://gist.github.com/traumverloren/fa5c30056319992c4dab)

- 每次打开终端都是用rbenv中的ruby
```bash
echo 'eval "$(~/.rbenv/bin/rbenv init - zsh)"' >> ~/.zshrc
```

- 碰到xcode报ruby版本不一致的问题 用下面代码创建符号连接
```
sudo ln -s `which ruby` /usr/local/bin/ruby
```

## rbenv安装ruby问题
```bash
# 描述
rbenv install 3.0.0

# error
-> ./configure "--prefix=$HOME/.rbenv/versions/3.0.0" --with-openssl-dir=/opt/homebrew/opt/openssl@1.1 --enable-shared --with-readline-dir=/opt/homebrew/opt/readline --with-libyaml-dir=/opt/homebrew/opt/libyaml --with-gmp-dir=/opt/homebrew/opt/gmp --with-ext=openssl,psych,+
-> make -j 8

BUILD FAILED (macOS 14.2.1 on arm64 using ruby-build 20240612)
```

如果安装3.2版本以下的ruby，则需要openssl@1.1，而不是openssl@3，但是如果不指定则不会寻找到，从而报错，解决方案：

1. 确保依赖安装正确

确保所有必要的依赖已经正确安装：
```bash
brew install openssl@1.1 readline libyaml gmp
```

2. 配置环境变量

设置环境变量以确保编译时能正确找到这些依赖：
```bash
export LDFLAGS="-L/opt/homebrew/opt/openssl@1.1/lib -L/opt/homebrew/opt/readline/lib -L/opt/homebrew/opt/libyaml/lib -L/opt/homebrew/opt/gmp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openssl@1.1/include -I/opt/homebrew/opt/readline/include -I/opt/homebrew/opt/libyaml/include -I/opt/homebrew/opt/gmp/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@1.1/lib/pkgconfig:/opt/homebrew/opt/readline/lib/pkgconfig:/opt/homebrew/opt/libyaml/lib/pkgconfig:/opt/homebrew/opt/gmp/lib/pkgconfig"
```