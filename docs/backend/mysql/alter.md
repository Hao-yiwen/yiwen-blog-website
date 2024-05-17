# Alter操作符

在SQL中，ALTER命令用于修改现有的数据库结构。这包括对表格、索引、视图等的修改。常见的ALTER操作包括：

1. 添加、删除或修改表格的列：
    - 添加列：ALTER TABLE 表名 ADD 列名 数据类型;
    - 删除列：ALTER TABLE 表名 DROP COLUMN 列名;
    - 修改列：ALTER TABLE 表名 MODIFY 列名 新数据类型;

2.  修改表格的属性：如更改存储引擎或字符集。
    - 更改字符集：ALTER TABLE 表名 CONVERT TO CHARACTER SET 字符集;

3. 管理索引：添加、删除或修改表格的索引。
    - 添加索引：ALTER TABLE 表名 ADD INDEX 索引名 (列名);

4. 修改约束：如添加或删除主键和外键。
    - 添加外键：ALTER TABLE 表名 ADD FOREIGN KEY (列名) REFERENCES 另一表名(列名);

## 使用实例

对现有的表添加主键、外键、默认值和自增属性通常涉及以下步骤：

1. 添加主键：

```sql
ALTER TABLE 表名 ADD PRIMARY KEY (列名);
```
如果表中已有数据，确保该列中没有重复值。

2. 添加外键：

```sql
ALTER TABLE 表名 ADD FOREIGN KEY (列名) REFERENCES 另一表名(对应列名);
```
确保参照的列在关联的表中存在。

3. 设置默认值：

```sql
ALTER TABLE 表名 ALTER COLUMN 列名 SET DEFAULT 默认值;
```
4. 设置自增：

```sql
ALTER TABLE 表名 MODIFY COLUMN 列名 数据类型 AUTO_INCREMENT;
```
通常用于整型列，如ID。