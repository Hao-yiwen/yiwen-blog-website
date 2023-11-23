---
sidebar_position: 3
---

# MongoDB

MongoDB 是一个流行的开源 NoSQL 数据库，由 MongoDB Inc. 开发。它是一个文档数据库，意味着它以文档的形式存储数据，这些文档的格式类似于 JSON。

MongoDB 在可扩展性、性能和灵活性方面表现出色，是开发现代应用程序的热门选择之一。以下是 MongoDB 的一些关键特点：

## 优点

### 文档导向
MongoDB 存储的数据单元是文档，这些文档组织成集合。文档由字段（key）和值（value）对组成，类似于 JSON 对象。

文档可以包含不同类型的数据，如字符串、数字、布尔值、数组，甚至嵌套文档。

### 查询语言
MongoDB 提供了强大的查询语言，允许您执行各种复杂的查询操作，包括文档字段的过滤、文档的排序和限制返回结果数量等。

它还支持聚合操作，允许您进行数据处理和分析。

### 索引
为了提高查询效率，MongoDB 支持对文档中的字段建立索引。

索引可以显著提高查询性能，特别是在处理大量数据时。

### 应用场景
MongoDB 适合需要处理大量数据且数据结构多变的应用程序，如内容管理系统、电子商务网站、数据仓库和大数据应用。

MongoDB 的这些特性使它成为当今应用程序开发中广泛使用的数据库之一，特别是在需要快速迭代和处理非结构化或半结构化数据的场景中。

## 安装

1. [mongoDB在Centos安装](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-red-hat/)
2. 使用navicat链接使用

## 使用

```js
const express = require('express');
const config = require('./config.json');
const {MongoClient} = require('mongodb');
const app = express();
app.use(express.json());

const url = config.mongodb.url;
const dbName = config.mongodb.dbName;

const client = new MongoClient(url, { useUnifiedTopology: true });

app.get('/', (req, res) => {
    res.send('Hello, World!');
});

app.post('/add', async (req, res) => {
    const {name, age} = req.body;
    try {
        await client.connect();
        const db = client.db(dbName);
        const collection = db.collection('users');
        const result = await collection.insertOne({name, age});
        res.json(result);
    } catch (err) {
        console.log(err);
    }
});

app.get('/users', async (req, res) => {
    try {
        await client.connect();
        const db = client.db(dbName);
        const collection = db.collection('users');
        const result = await collection.find({}).toArray();
        res.json(result);
    } catch (err) {
        console.log(err);
    }
});

app.get('/users/age', async (req, res) => {
    const {age} = req.query;
    try {
        await client.connect();
        const db = client.db(dbName);
        const collection = db.collection('users');
        const result = await collection.find({ age: { $gt: 25 } }).toArray();
        res.json(result);
    } catch (err) {
        console.log(err);
    }
});

const PORT = 3000;

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
```

## 和mysql的比较

### mongoDB

1. 灵活的数据模型：MongoDB 使用文档型数据模型，无需预定义模式，适合快速迭代和处理多变的数据结构。
2. 水平扩展性：通过分片（Sharding），MongoDB 支持大规模的水平扩展。
3. 高效的处理大量数据：适合大数据应用和实时分析。
4. 强大的查询语言：支持复杂的查询和聚合操作。
5. 高可用性和自动故障转移：通过复制集实现。
6. 适合非结构化和半结构化数据：如 JSON 或 BSON 格式。

### mysql

1. 成熟的技术：作为关系型数据库，MySQL 有着多年的发展和广泛的应用，技术成熟。
2. 严格的数据模式：有助于确保数据的完整性和一致性。
3. 强大的事务支持：支持复杂的事务处理和锁定机制。
4. 高效的 JOIN 操作：在处理关系型数据时表现出色。
5. 广泛的社区和工具支持：有大量的工具和社区支持。

