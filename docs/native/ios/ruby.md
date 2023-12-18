# ruby安装踩坑

## openssl安装

`20231218`因为`openssl`版本问题，导致`rvm`管理ruby版本失败，总结经验发现`openssl`版本需要`1.1.1`

## 使用rvm安装ruby@2.7.x版本

```bash
export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@1.1/lib/pkgconfig"

rvm reinstall 2.7.2 --with-openssl-dir=/opt/homebrew/opt/openssl@1.1
```
