---
title: Pandas 常用操作速查手册
sidebar_label: Pandas 速查手册
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# Pandas 常用操作速查手册

## 📦 导入和创建

```python
import pandas as pd
import numpy as np

# 创建 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd'],
    'C': [1.1, 2.2, 3.3, 4.4]
})

# 创建 Series
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# 从字典创建(指定索引)
df = pd.DataFrame({
    'col1': [1, 2],
    'col2': [3, 4]
}, index=['row1', 'row2'])
```

---

## 📂 数据读取和保存

```python
# CSV 文件
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', encoding='utf-8')  # 指定编码
df = pd.read_csv('file.csv', sep='\t')          # 指定分隔符
df.to_csv('output.csv', index=False)            # 保存(不保存索引)

# Excel 文件
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', index=False)

# JSON 文件
df = pd.read_json('file.json')
df.to_json('output.json')

# SQL 数据库
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df.to_sql('table_name', conn, if_exists='replace')
```

---

## 👀 数据查看

```python
# 基本信息
df.head()           # 前5行
df.head(10)         # 前10行
df.tail()           # 后5行
df.sample(5)        # 随机5行

# 数据概览
df.shape            # (行数, 列数)
df.info()           # 数据类型、非空值数量
df.describe()       # 数值列的统计摘要
df.dtypes           # 每列的数据类型
df.columns          # 列名
df.index            # 索引

# 统计信息
df.count()          # 非空值数量
df.mean()           # 平均值
df.median()         # 中位数
df.std()            # 标准差
df.min()            # 最小值
df.max()            # 最大值
df.sum()            # 求和
df.value_counts()   # Series 的值计数
```

---

## 🔍 数据选择和索引

### 选择列

```python
df['A']                 # 选择单列(返回 Series)
df[['A', 'B']]          # 选择多列(返回 DataFrame)
df.A                    # 通过属性访问(不推荐用于有空格的列名)
```

### 选择行

```python
# 基于位置 - iloc
df.iloc[0]              # 第一行
df.iloc[0:3]            # 前3行
df.iloc[[0, 2, 4]]      # 特定行
df.iloc[:, 0:2]         # 所有行,前2列

# 基于标签 - loc
df.loc[0]               # 索引为0的行
df.loc[0:3]             # 索引0到3(包括3)
df.loc[:, 'A':'C']      # 所有行,列A到C
df.loc[df['A'] > 2]     # 条件筛选
```

### 条件筛选

```python
# 单条件
df[df['A'] > 2]
df[df['B'] == 'a']

# 多条件(与)
df[(df['A'] > 2) & (df['B'] == 'a')]

# 多条件(或)
df[(df['A'] > 2) | (df['B'] == 'a')]

# 取反
df[~(df['A'] > 2)]

# isin 方法
df[df['B'].isin(['a', 'b'])]

# 字符串包含
df[df['B'].str.contains('a')]

# 查询语法(更简洁)
df.query('A > 2 and B == "a"')
```

---

## 🔧 数据清洗

### 缺失值处理

```python
# 检查缺失值
df.isnull()             # 返回布尔 DataFrame
df.isnull().sum()       # 每列缺失值数量
df.isnull().any()       # 每列是否有缺失值

# 删除缺失值
df.dropna()             # 删除任何包含缺失值的行
df.dropna(axis=1)       # 删除任何包含缺失值的列
df.dropna(how='all')    # 删除全为缺失值的行
df.dropna(thresh=2)     # 保留至少有2个非空值的行

# 填充缺失值
df.fillna(0)            # 用0填充
df.fillna(method='ffill')   # 前向填充
df.fillna(method='bfill')   # 后向填充
df['A'].fillna(df['A'].mean())  # 用平均值填充
```

### 重复值处理

```python
# 检查重复
df.duplicated()         # 返回布尔 Series
df.duplicated().sum()   # 重复行数量

# 删除重复
df.drop_duplicates()    # 删除重复行(保留第一次出现)
df.drop_duplicates(keep='last')     # 保留最后一次出现
df.drop_duplicates(subset=['A'])    # 基于特定列判断重复
```

### 数据类型转换

```python
# 转换单列
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(str)
df['C'] = pd.to_numeric(df['C'], errors='coerce')  # 无法转换的设为 NaN

# 转换多列
df = df.astype({'A': int, 'B': str})

# 日期时间转换
df['date'] = pd.to_datetime(df['date'])
```

---

## ✏️ 数据修改

### 添加/删除列

```python
# 添加列
df['D'] = [5, 6, 7, 8]          # 直接赋值
df['E'] = df['A'] + df['C']     # 计算得到
df.insert(1, 'F', [9, 10, 11, 12])  # 在指定位置插入

# 删除列
df.drop('D', axis=1, inplace=True)      # 删除单列
df.drop(['D', 'E'], axis=1, inplace=True)   # 删除多列
del df['F']                             # 直接删除
```

### 重命名

```python
# 重命名列
df.rename(columns={'A': 'new_A', 'B': 'new_B'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']   # 重命名所有列

# 重命名索引
df.rename(index={0: 'row1', 1: 'row2'}, inplace=True)
```

### 修改值

```python
# 修改单个值
df.loc[0, 'A'] = 100
df.iloc[0, 0] = 100

# 修改整列
df['A'] = 0
df['A'] = df['A'] * 2

# 条件修改
df.loc[df['A'] > 2, 'B'] = 'high'

# 使用 replace
df['B'].replace('a', 'new_a', inplace=True)
df.replace({'a': 'new_a', 'b': 'new_b'}, inplace=True)
```

---

## 🔄 数据转换

### Apply 函数

```python
# 对列应用函数
df['A'].apply(lambda x: x * 2)
df['A'].apply(np.sqrt)

# 对 DataFrame 应用函数
df.apply(lambda x: x.max() - x.min())   # 对每列
df.apply(lambda x: x.max() - x.min(), axis=1)  # 对每行

# map(仅用于 Series)
df['B'].map({'a': 1, 'b': 2, 'c': 3})

# applymap(对每个元素) - 已弃用,使用 map
df[['A', 'C']].map(lambda x: x * 2)
```

### 独热编码(One-Hot Encoding)

```python
# 基本用法 - 将类别型变量转换为 0/1 虚拟变量
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'M'],
    'price': [10, 20, 30, 15]
})

# 对所有非数值列自动编码
pd.get_dummies(df)
# 结果: price, color_blue, color_green, color_red, size_L, size_M, size_S

# 只对特定列编码
pd.get_dummies(df, columns=['color'])

# 指定列名前缀
pd.get_dummies(df, prefix={'color': '颜色', 'size': '尺寸'})

# 删除第一列(避免多重共线性,用于线性回归)
pd.get_dummies(df, drop_first=True)

# 处理缺失值
pd.get_dummies(df, dummy_na=True)

# 实际应用:训练集和测试集一致性编码
all_data = pd.concat([train_df, test_df])
all_encoded = pd.get_dummies(all_data, columns=['category_col'])
train_encoded = all_encoded[:len(train_df)]
test_encoded = all_encoded[len(train_df):]
```

### 排序

```python
# 按值排序
df.sort_values('A')                     # 升序
df.sort_values('A', ascending=False)    # 降序
df.sort_values(['A', 'B'])              # 多列排序
df.sort_values('A', inplace=True)       # 原地排序

# 按索引排序
df.sort_index()
```

### 分组操作

```python
# 基本分组
df.groupby('B').mean()          # 按B列分组,计算平均值
df.groupby('B').sum()           # 求和
df.groupby('B').count()         # 计数
df.groupby('B').size()          # 每组大小

# 多列分组
df.groupby(['B', 'C']).mean()

# 聚合多个统计量
df.groupby('B').agg(['mean', 'sum', 'count'])

# 对不同列应用不同聚合
df.groupby('B').agg({
    'A': 'mean',
    'C': ['sum', 'max']
})

# 自定义聚合函数
df.groupby('B').agg(lambda x: x.max() - x.min())
```

---

## 🔗 数据合并

### Concat(拼接)

```python
# 垂直拼接(行)
pd.concat([df1, df2], axis=0)
pd.concat([df1, df2], ignore_index=True)  # 重置索引

# 水平拼接(列)
pd.concat([df1, df2], axis=1)
```

### Merge(合并)

```python
# 内连接(默认)
pd.merge(df1, df2, on='key')

# 左连接
pd.merge(df1, df2, on='key', how='left')

# 右连接
pd.merge(df1, df2, on='key', how='right')

# 外连接
pd.merge(df1, df2, on='key', how='outer')

# 多个键
pd.merge(df1, df2, on=['key1', 'key2'])

# 不同列名的键
pd.merge(df1, df2, left_on='key1', right_on='key2')
```

### Join(连接)

```python
# 基于索引连接
df1.join(df2)
df1.join(df2, how='left')
df1.join(df2, lsuffix='_left', rsuffix='_right')  # 处理重复列名
```

---

## 📊 数据透视和重塑

### Pivot(透视)

```python
# 创建透视表
df.pivot(index='date', columns='category', values='value')

# 透视表(支持聚合)
df.pivot_table(
    values='value',
    index='date',
    columns='category',
    aggfunc='mean'
)
```

### Melt(逆透视)

```python
# 宽格式转长格式
pd.melt(df, id_vars=['id'], value_vars=['A', 'B', 'C'])
```

### Stack/Unstack

```python
# Stack:列转行
df.stack()

# Unstack:行转列
df.unstack()
```

---

## 📈 字符串操作

```python
# 字符串方法(需要 .str 访问器)
df['B'].str.upper()             # 转大写
df['B'].str.lower()             # 转小写
df['B'].str.strip()             # 去除空格
df['B'].str.replace('a', 'A')   # 替换
df['B'].str.contains('a')       # 是否包含
df['B'].str.startswith('a')     # 是否以...开头
df['B'].str.endswith('a')       # 是否以...结尾
df['B'].str.split('_')          # 分割
df['B'].str.len()               # 字符串长度
df['B'].str[0]                  # 取第一个字符
df['B'].str[:3]                 # 切片
```

---

## 📅 日期时间操作

```python
# 创建日期范围
dates = pd.date_range('2024-01-01', periods=10, freq='D')

# 日期时间属性
df['date'].dt.year              # 年
df['date'].dt.month             # 月
df['date'].dt.day               # 日
df['date'].dt.dayofweek         # 星期几(0=周一)
df['date'].dt.hour              # 小时
df['date'].dt.minute            # 分钟

# 日期计算
df['date'] + pd.Timedelta(days=7)       # 加7天
df['date'] - pd.Timedelta(hours=2)      # 减2小时

# 设置日期为索引
df.set_index('date', inplace=True)

# 按时间筛选
df['2024']                      # 2024年的数据
df['2024-01']                   # 2024年1月的数据
```

---

## 🎯 实用技巧

### 链式操作

```python
result = (df
    .dropna()
    .query('A > 2')
    .groupby('B')
    .mean()
    .sort_values('A', ascending=False)
)
```

### 设置显示选项

```python
pd.set_option('display.max_rows', 100)      # 最大显示行数
pd.set_option('display.max_columns', 20)    # 最大显示列数
pd.set_option('display.width', 1000)        # 显示宽度
pd.set_option('display.precision', 2)       # 小数精度
pd.reset_option('all')                      # 重置所有选项
```

### 性能优化

```python
# 使用分类类型节省内存
df['B'] = df['B'].astype('category')

# 读取大文件时分块
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# 使用 eval 进行向量化计算(更快)
df.eval('D = A + C', inplace=True)
```

---

## 💡 常见数据清洗流程

```python
# 完整的数据清洗示例
df = (df
    .drop_duplicates()                          # 删除重复
    .dropna(subset=['important_col'])           # 删除关键列缺失值
    .fillna({'A': 0, 'B': 'unknown'})          # 填充其他缺失值
    .assign(                                     # 添加/修改列
        D=lambda x: x['A'] * 2,
        E=lambda x: x['B'].str.upper()
    )
    .query('A > 0')                             # 筛选
    .reset_index(drop=True)                     # 重置索引
)

# 机器学习预处理流程
# 1. 分离数值和类别特征
numeric_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=['object']).columns

# 2. 处理数值特征
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# 3. 处理类别特征并编码
df[categorical_features] = df[categorical_features].fillna('missing')
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
```

---

## 🔍 快速查找和索引

```python
# 查找特定值的位置
df[df['A'] == 5].index

# 按条件获取第一行/最后一行
df[df['A'] > 2].head(1)
df[df['A'] > 2].iloc[0]

# 重置索引
df.reset_index(drop=True, inplace=True)

# 设置索引
df.set_index('A', inplace=True)

# 多级索引
df.set_index(['A', 'B'], inplace=True)
```

---

## 📝 小抄速查

```python
# 最常用的操作
df.head()                       # 查看前几行
df.info()                       # 数据概览
df.describe()                   # 统计摘要
df[df['col'] > 5]              # 条件筛选
df.groupby('col').mean()       # 分组聚合
df.sort_values('col')          # 排序
df.dropna()                    # 删除缺失值
df.fillna(0)                   # 填充缺失值
pd.get_dummies(df)             # 独热编码(类别变量转数值)
pd.merge(df1, df2, on='key')   # 合并
df.to_csv('file.csv')          # 保存
```

---

**这份速查手册涵盖了 pandas 95% 的日常使用场景,建议收藏!** 📚

---

*参考资源:*
- [Pandas 官方文档](https://pandas.pydata.org/docs/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
