---
title: redis-cli 工具与常用命令速查
sidebar_label: redis-cli 与常用命令
sidebar_position: 2
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# redis-cli 工具与常用命令速查

`redis-cli` 是 Redis 官方命令行客户端，支持交互式命令执行、远程连接、管道批处理（Pipeline）与实时监控。

## 一、连接与基础操作

```bash
# 连接本机默认实例 (127.0.0.1:6379)
redis-cli

# 连接指定远程主机与端口
redis-cli -h 127.0.0.1 -p 6379

# 密码鉴权（交互模式下）
AUTH <password>

# 批量执行文件中的命令
redis-cli < my_commands.txt

# 退出客户端
quit
```

### 高级调试能力

- **实时监控**：通过 `MONITOR` 实时打印服务端收到的命令流（生产环境慎用，有性能开销）。
- **管道批处理**：通过 `--pipe` 或 Pipeline 批量发送命令，减少网络往返 RTT。

---

## 二、常用命令分类速查

### 1. 通用键（Key）管理

| 命令 | 说明 |
| :--- | :--- |
| `KEYS pattern` | 查找匹配模式的键（生产环境大数据量请改用 `SCAN`） |
| `EXISTS key` | 检查键是否存在 |
| `DEL key` | 删除指定键 |
| `EXPIRE key seconds` | 设置过期时间（秒） |
| `TTL key` | 查看键的剩余存活秒数 |

### 2. 字符串（String）

- `SET key value`：设置字符串值
- `GET key`：获取字符串值
- `INCR key` / `DECR key`：将整数值原子自增 / 自减 1
- `APPEND key value`：追加内容到原字符串末尾

### 3. 列表（List）

- `LPUSH key value` / `RPUSH key value`：从列表头部 / 尾部插入元素
- `LPOP key` / `RPOP key`：移除并返回列表头 / 尾元素
- `LRANGE key start stop`：获取指定区间内的元素

### 4. 集合（Set）与有序集合（Sorted Set）

- **Set**：`SADD`（添加）、`SMEMBERS`（获取全部成员）、`SISMEMBER`（判断存在）、`SREM`（移除）
- **ZSet**：`ZADD key score member`（带分值添加）、`ZRANGE` / `ZREVRANGE`（按分值正序/倒序区间查询）、`ZREM`（移除）

### 5. 哈希（Hash）

- `HSET key field value`：设置哈希表字段值
- `HGET key field`：读取单个字段值
- `HGETALL key`：获取所有字段与值
- `HDEL key field`：删除一个或多个字段

### 6. 扩展结构与发布订阅

- **Geospatial**：`GEOADD key longitude latitude member`、`GEODIST key m1 m2 [unit]`
- **HyperLogLog**：`PFADD key element`、`PFCOUNT key`
- **Pub/Sub**：`PUBLISH channel message`、`SUBSCRIBE channel`
