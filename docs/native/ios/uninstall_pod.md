# 卸载pod

1. 查找所有相关pod依赖
```bash
gem list --local | grep cocoapods
```

2. 删除列出的所有cocoapods相关依赖,例如

```bash
gem install cocoapods-search
```