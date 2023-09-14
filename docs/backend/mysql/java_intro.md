---
sidebar_position: 1
---

# 修改密码

## 登录到MySQL服务器

```bash
mysql -u root -p
```

## 选择mysql数据库

```sql
USE mysql;
```

## 修改密码

```sql
ALTER USER 'username'@'host' IDENTIFIED BY 'new_password';
```

## 刷新权限

```sql
FLUSH PRIVILEGES;
```
