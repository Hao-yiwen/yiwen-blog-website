---
title: "常用redis-cli命令"
sidebar_label: "常用redis-cli命令"
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 常用redis-cli命令

Redis 提供了大量的命令用于处理各种数据类型和任务。以下是一些常用的 Redis 命令：

## 通用命令

-   KEYS pattern：查找所有符合给定模式的键。
-   EXISTS key：检查给定键是否存在。
-   DEL key：删除一个键。
-   EXPIRE key seconds：为键设置过期时间。
-   TTL key：查看键的剩余生存时间。

## 字符串（String）命令

-   SET key value：设置字符串值。
-   GET key：获取字符串值。
-   INCR key：将键的整数值增加1。
-   DECR key：将键的整数值减少1。
-   APPEND key value：将值追加到原来值的末尾。

## 列表（List）命令

-   LPUSH key value：将一个或多个值插入到列表头部。
-   RPUSH key value：将一个或多个值插入到列表尾部。
-   LPOP key：移除并返回列表的第一个元素。
-   RPOP key：移除并返回列表的最后一个元素。
-   LRANGE key start stop：获取列表指定范围内的元素。

## 集合（Set）命令

-   SADD key member：向集合添加一个或多个成员。
-   SMEMBERS key：获取集合中的所有成员。
-   SISMEMBER key member：判断成员元素是否是集合的成员。
-   SREM key member：移除集合中一个或多个成员。

## 有序集合（Sorted Set）命令

-   ZADD key score member：向有序集合添加一个或多个成员。
-   ZRANGE key start stop [WITHSCORES]：通过索引区间返回有序集合成指定区间内的成员。
-   ZREVRANGE key start stop [WITHSCORES]：返回有序集中指定区间内的成员，通过索引，分数从高到低。
-   ZREM key member：移除有序集合中的一个或多个成员。

## 哈希（Hash）命令

-   HSET key field value：向哈希表中添加字段。
-   HGET key field：获取哈希表中的字段值。
-   HGETALL key：获取在哈希表中指定 key 的所有字段和值。
-   HDEL key field：删除哈希表 key 中的一个或多个指定字段。

## 其他数据结构的命令

-   GEOADD key longitude latitude member：添加地理空间位置。
-   GEODIST key member1 member2 [unit]：返回两个地理空间位置之间的距离。
-   PFADD key element：添加指定元素到 HyperLogLog 中。
-   PFCOUNT key：返回 HyperLogLog 的近似基数估算。

## 发布/订阅命令

-   PUBLISH channel message：向指定频道发布消息。
-   SUBSCRIBE channel：订阅给定的一个或多个频道的信息。
