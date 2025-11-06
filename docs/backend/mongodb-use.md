---
title: MongoDB使用
sidebar_label: MongoDB使用
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# MongoDB使用

## 数据库操作

1. 查看所有数据库:
```bash
show dbs
```
2. 切换或创建数据库 (如果不存在):
```bash
use [数据库名称]
```
3. 删除当前数据库:
```js
db.dropDatabase()
```

### 集合操作
1. 查看当前数据库的所有集合:
```bash
show collections
```

2. 创建集合:
```js
db.createCollection("集合名称")
```

3. 删除集合:

```js
db.集合名称.drop()
```

### 文档操作
1. 插入文档:
```js
db.集合名称.insert({字段1: 值1, 字段2: 值2})
```
2. 查找文档:
```js
db.集合名称.find({查询条件})
```
3. 更新文档:
```js
db.集合名称.update({查询条件}, {$set: {字段1: 新值}})
```
4. 删除文档:
```js
db.集合名称.remove({查询条件})
```

### 用户管理
1. 创建用户:
```js
db.createUser({user: "用户名", pwd: "密码", roles: [{role: "角色", db: "数据库"}]})
```
2. 列出所有用户:
```bash
show users
```
3. 修改用户密码:
```js
db.changeUserPassword("用户名", "新密码")
```
### 索引管理
1. 创建索引:
```js
db.集合名称.createIndex({字段: 1}) // 1 为升序，-1 为降序
```
2. 查看集合的所有索引:
```js
db.集合名称.getIndexes()
```
3. 删除索引:
```js
db.集合名称.dropIndex("索引名称")
```

### 其他有用的命令
1. 查看服务器状态:
```js
db.serverStatus()
```
2. 查看当前数据库状态:
```js
db.stats()
```
3. 查看集合的存储空间使用情况:
```js
db.集合名称.stats()
```