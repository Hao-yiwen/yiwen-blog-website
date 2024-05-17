# mysql使用

- 创建数据库：
```sql
CREATE DATABASE database_name;
```

- 删除数据库：
```sql
DROP DATABASE database_name;
```

- 切换数据库
```sql
USE database
```

- 创建表格：
```sql
CREATE TABLE table_name (column1 datatype, column2 datatype, ...);
```

- 创建有主键的表格
在创建表时，您可以通过PRIMARY KEY关键字指定一个或多个列作为主键
```
CREATE TABLE Persons (
    PersonID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    PRIMARY KEY (PersonID)
);
```

- 创建有外键的表格
```sql
CREATE TABLE Orders (
    OrderID int NOT NULL,
    OrderNumber int NOT NULL,
    PersonID int,
    PRIMARY KEY (OrderID),
    FOREIGN KEY (PersonID) REFERENCES Persons(PersonID)
);
```

- 删除表格：
```sql
DROP TABLE table_name;
```

- 插入数据：
```sql
INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...);
```

- 查询数据：
```sql
# 查询所有数据
SELECT * FROM table_name;

# 查询部分数据
SELECT employees.name, departments.department_name FROM employees

# 内连接
SELECT employees.name, departments.department_name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;

# 左外连接
SELECT employees.name, departments.department_name
FROM employees
LEFT OUTER JOIN departments ON employees.department_id = departments.id;

# 筛选（Filtering）：使用WHERE子句筛选数据。

SELECT * FROM Orders WHERE customerID = 123 AND amount > 100;

# 聚合（Aggregation）：使用GROUP BY和聚合函数（如SUM、COUNT等）。

SELECT customerID, COUNT(*) FROM Orders GROUP BY customerID;

# 连接（Joining）：使用各种JOIN操作将数据与其他表连接。

SELECT Orders.orderID, Customers.name FROM Orders INNER JOIN Customers ON Orders.customerID = Customers.customerID;

# 限制结果数量（Limiting Results）：使用LIMIT限制返回的记录数。

SELECT * FROM Orders LIMIT 10;
```

- 关联查询
```sql
SELECT employees.name, departments.department_name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;
```

- 表格新增属性
```sql
ALTER TABLE table_name ADD column1 datatype;

# 示例
ALTER TABLE employees ADD birthdate DATE;
```

- 更新数据：
```sql
UPDATE table_name SET column1 = value1, column2 = value2, ... WHERE condition;
```

- 删除数据：
```sql
DELETE FROM table_name WHERE condition;
```

- 连接表格：
```sql
SELECT columns FROM table1 JOIN table2 ON table1.column = table2.column;
```

- 创建索引：
```sql
CREATE INDEX index_name ON table_name (column1, column2, ...);
```

- 合并查询结果

    - UNION：合并两个或多个查询结果，自动去除重复的行。
    - UNION ALL：类似于UNION，但包括所有重复行，不进行去重。
    - EXCEPT：返回第一个查询的结果，但排除在第二个查询中出现的行（mysql不支持）。
    - INTERSECT：返回两个查询共有的行，即仅出现在两个查询结果中的行（mysql不支持）。
