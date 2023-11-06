---
sidebar_position: 4
---

# sqlite

## 介绍

SQLite 是一个流行的嵌入式数据库管理系统，它不同于典型的客户端-服务器数据库系统。以下是SQLite的一些关键特性和信息：

1. 嵌入式 SQL 数据库引擎:

SQLite 是一个在应用程序内部运行的零配置、无需服务器和无需安装的数据库引擎。2. 轻量级:

它非常小巧，适合任何规模的系统，从嵌入式设备到智能手机再到台式机和服务器。3. 自给自足:

SQLite 不需要一个独立的服务器进程或系统来操作，数据库就是一个文件。4. 跨平台:

它提供了跨平台的支持，在大多数操作系统上都能够运行。5. 事务性:

SQLite 支持 SQL 标准的事务，是 ACID 兼容的，这意味着即使在系统崩溃或电源失效的情况下，所有的事务也都是安全的。

## 在flutter中使用

```dart
import 'package:flutter/material.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'dart:async';

void main() async => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter SQLite Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Database database;

  @override
  void initState() {
    super.initState();
    initializeDatabase();
  }

  Future<void> initializeDatabase() async {
    // 获取数据库路径
    String databasePath = await getDatabasesPath();
    String path = join(databasePath, 'my_database.db');

    // 打开/创建数据库
    database = await openDatabase(path, version: 1, onCreate: (Database db, int version) async {
      // 创建表
      await db.execute('CREATE TABLE Test (id INTEGER PRIMARY KEY, name TEXT)');
    });

    // 插入数据
    await insertTestData();

    // 查询数据
    List<Map> list = await database.rawQuery('SELECT * FROM Test');
    print(list);
  }

  Future<void> insertTestData() async {
    await database.transaction((txn) async {
      int id1 = await txn.rawInsert(
        'INSERT INTO Test(name) VALUES("lalala")'
      );
      print('inserted1: $id1');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flutter SQLite Demo'),
      ),
      body: Center(
        child: Text('Check console for SQL operations results.'),
      ),
    );
  }

  @override
  void dispose() {
    database.close();
    super.dispose();
  }
}

```

## QA

## sqlite支持在macos， web使用吗？

Flutter 的 sqflite 插件支持在 iOS 和 Android 上使用 SQLite 数据库，但它不支持 macOS 和 Web 平台。在 Flutter 中，不同平台的数据存储需求和支持情况可能有所不同。
