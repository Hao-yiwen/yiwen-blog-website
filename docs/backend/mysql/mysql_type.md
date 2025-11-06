---
title: mysql常见类型
sidebar_label: mysql常见类型
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# mysql常见类型

MySQL中常见的数据类型包括：

## 数值类型：

-   整型：如INT, SMALLINT, TINYINT, MEDIUMINT, BIGINT
-   浮点型和定点型：如FLOAT, DOUBLE, DECIMAL

## 字符串类型：

-   字符串：如VARCHAR, CHAR
-   文本：如TEXT, TINYTEXT, MEDIUMTEXT, LONGTEXT

## 日期和时间类型：

-   如DATE, TIME, DATETIME, TIMESTAMP, YEAR

## 二进制类型：

-   如BINARY, VARBINARY, BLOB, TINYBLOB, MEDIUMBLOB, LONGBLOB

## 逻辑类型：

-   如BOOLEAN或BOOL（实际上是TINYINT(1)的别名）

## 应用场景

1. 数值类型：

-   INT：存储没有小数的整数，如用户年龄或产品数量。
-   SMALLINT：存储较小范围的整数，例如评分系统的评分。
-   BIGINT：作为一个超大数据集的主键或唯一标识符，如用户ID、订单号等，尤其在大数据应用或需要非常高容量的系统中，或者存储那些可能超出标准整型范围的数值，例如全球用户数量、某些计算结果等
-   FLOAT, DOUBLE：存储带小数的数值，如商品价格、测量数据。
-   DECIMAL：用于存储需要精确计算的数值，常用于财务数据。

2. 字符串类型：

-   VARCHAR：用于存储可变长度的字符串，如姓名、电子邮件地址。
-   CHAR：用于存储固定长度的字符串，例如性别、国家代码。
-   TEXT：用于存储长文本，如文章、评论或日志。

3. 日期和时间类型：

-   DATE：仅存储日期，适用于生日、纪念日等。
-   TIME：仅存储时间，适用于事件或日程的时间。
-   DATETIME, TIMESTAMP：存储日期和时间，用于时间戳、记录创建或修改时间。

4. 二进制类型：

-   BINARY, VARBINARY：存储固定或可变长度的二进制数据，如加密数据。
-   BLOB：存储大型二进制对象，如图片、视频或音频文件。

5. 逻辑类型：

-   BOOLEAN：存储布尔值（真/假），用于表示开关状态、是否选项等。
