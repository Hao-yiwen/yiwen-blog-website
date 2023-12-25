# INNER JOIN和OUTER JOIN

## 介绍

1. INNER JOIN和OUTER JOIN是SQL中两种不同的表关联方式：

2. INNER JOIN：仅返回两个表中匹配的记录。如果表A和表B之间做INNER JOIN，结果中只包含A和B都有的那部分数据。OUTER JOIN：可以分为三种：LEFT OUTER JOIN、RIGHT OUTER JOIN和FULL OUTER JOIN。
    - LEFT OUTER JOIN（左连接）：返回左表（JOIN语句前的表）的所有记录，即使在右表中没有匹配的记录。右表中没有匹配的部分会显示为NULL。
    - RIGHT OUTER JOIN（右连接）：返回右表的所有记录，即使在左表中没有匹配的记录。左表中没有匹配的部分会显示为NULL。
    - FULL OUTER JOIN（全外连接[mysql中暂不支持]）：返回两个表中所有的记录，不论它们之间是否匹配。不匹配的部分会显示为NULL。

## 示例

表结构

```sql
-- 假设的表结构
employees: id, name, department_id
departments: id, department_name
```

### INNER JOIN

```sql
SELECT employees.name, departments.department_name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;
```
将返回在employees和departments表中都有匹配的记录。

### OUTER JOIN

```sql
SELECT employees.name, departments.department_name
FROM employees
LEFT OUTER JOIN departments ON employees.department_id = departments.id;
```

这将返回所有employees记录和它们对应的departments记录。如果某个员工没有对应的部门，部门名将显示为NULL。

```sql
SELECT employees.name, departments.department_name
FROM employees
RIGHT JOIN departments ON employees.department_id = departments.id;
```

这将返回所有employees记录和它们对应的departments记录。如果某个部门没有对应的员工，员工名将显示为NULL。

## 使用场景

在数据库查询中，INNER JOIN和OUTER JOIN的使用场景通常取决于你需要从关联表中检索哪些数据：

1. INNER JOIN：适用于你只对两个表中都存在匹配的数据感兴趣的情况。它只返回在两个表中都有相匹配记录的结果。例如，你只想要那些在员工表和部门表中都有记录的员工信息。

2. OUTER JOIN：适用于你需要从一个表中获取所有记录，并从另一个表中获取匹配的记录（如果存在）的情况。如果第二个表中没有匹配记录，仍然会返回第一个表中的记录。

    - LEFT OUTER JOIN：返回左表的所有记录，以及与右表匹配的记录。
    - RIGHT OUTER JOIN：返回右表的所有记录，以及与左表匹配的记录。
    - FULL OUTER JOIN（不是所有数据库系统都支持）：返回两个表中的所有记录，不论它们之间是否匹配。

## 模拟器 FULL OUTER JOIN

:::info
`mysql`不支持`full join`,但是可以模拟实现
:::

```sql
SELECT employees.name, departments.department_name
FROM employees
LEFT JOIN departments ON employees.department_id = departments.id
UNION
SELECT employees.name, departments.department_name
FROM employees
RIGHT JOIN departments ON employees.department_id = departments.id;
```