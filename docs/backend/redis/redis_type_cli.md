---
title: Redis数据结构
sidebar_label: Redis数据结构
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Redis数据结构

## 字符串（String）

描述：最基本的类型，可以存储文本或二进制数据，常用于缓存、计数器等。
常用命令：SET, GET, INCR, DECR, APPEND。
示例场景：存储用户的邮箱地址，增加页面访问计数。

```bash
# 存储用户的最后登录时间
SET last_login_time_user123 "2022-09-01 10:00:00"

# 检索用户的最后登录时间
GET last_login_time_user123
```

## 列表（List）

描述：字符串列表，按插入顺序排序。可以用作简单的消息队列。
常用命令：LPUSH, RPUSH, LPOP, RPOP, LRANGE。
示例场景：存储最新消息的队列，实现时间线或活动日志。

```bash
# 将登录时间推入列表
LPUSH user123_login_times "2022-09-01 10:00:00"
LPUSH user123_login_times "2022-09-02 09:30:00"
LPUSH user123_login_times "2022-09-03 08:45:00"

# 获取最近三次登录时间
LRANGE user123_login_times 0 2
```

## 集合（Set）

描述：字符串的无序集合，主要支持成员的快速存取、添加、删除。
常用命令：SADD, SMEMBERS, SISMEMBER, SREM, SUNION。
示例场景：存储唯一元素的集合，如标签、好友列表。

```bash
# 添加兴趣爱好
SADD user123_hobbies "music" "travel" "photography"

# 获取用户的兴趣爱好
SMEMBERS user123_hobbies
```

## 有序集合（Sorted Set）

描述：不仅集合元素唯一，而且每个元素都关联一个浮点数分数，用于排序。
常用命令：ZADD, ZRANGE, ZREVRANGE, ZINCRBY, ZRANK。
示例场景：排行榜系统，如音乐排行榜、游戏分数排行。

```bash
# 添加玩家得分
ZADD game_scores 300 "player1"
ZADD game_scores 250 "player2"
ZADD game_scores 450 "player3"

# 获取得分最高的3名玩家
ZREVRANGE game_scores 0 2 WITHSCORES
```

## 哈希表（Hash）

描述：键值对集合，适用于存储对象。
常用命令：HSET, HGET, HMSET, HMGET, HGETALL。
示例场景：存储和管理对象数据，如用户属性。

```bash
# 存储用户资料
HSET user123_profile name "John Doe" age "30" country "USA"

# 检索用户资料
HGETALL user123_profile
```

## 位图（Bitmaps）

描述：通过位操作处理字符串值。适合存储布尔值或进行统计计算。
常用命令：SETBIT, GETBIT, BITCOUNT, BITOP。
示例场景：用户签到功能，统计在线用户数。

```bash
# 标记用户在某天登录
SETBIT user123_login_2022 1 1

# 检查用户是否在某天登录
GETBIT user123_login_2022 1
```

## 超日志（HyperLogLog）

描述：一种概率数据结构，用于高效计算集合的近似基数（不同元素的数量）。
常用命令：PFADD, PFCOUNT, PFMERGE。
示例场景：大规模数据的唯一计数，如统计网站访客数。

```bash
# 添加用户到 HyperLogLog
PFADD website_visitors user123 user124 user125

# 估算不同的访问者数量
PFCOUNT website_visitors
```

## 地理空间（Geospatial）

描述：用于存储地理位置信息并进行范围查询。
常用命令：GEOADD, GEODIST, GEORADIUS, GEOHASH。
示例场景：位置服务，如查找附近的商店或服务。

```bash
# 存储地点
GEOADD city_locations 116.405285 39.904989 "Beijing"
GEOADD city_locations 121.472644 31.231706 "Shanghai"

# 查询距离
GEODIST city_locations "Beijing" "Shanghai"
```

## 总结

Redis的多样化数据结构为各种场景下的数据存储和管理提供了强大的支持。选择合适的数据结构可以大大提高应用程序的性能和效率。在实际使用中，根据具体需求合理选择数据结构是非常重要的。
