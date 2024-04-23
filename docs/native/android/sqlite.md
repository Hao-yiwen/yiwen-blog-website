# Sqlite

## sqlite的基础数据类型

-   https://www.sqlite.org/datatype3.html

SQLite 是一个轻量级的数据库系统，它支持一套相对简单的数据类型。在 SQLite 中，所有值都以以下五种基础数据类型之一存储：

1. NULL：
    - 这个类型用于表示缺失的值或空值。
2. INTEGER：
    - 这种数据类型用于存储整数值。根据整数的大小，SQLite 可以动态调整存储空间，从 1 字节到 8 字节不等。
3. REAL：
    - 用于存储浮点数，通常是双精度64位IEEE浮点数。
4. TEXT：
    - 用于存储文本数据。SQLite 使用数据库编码（UTF-8、UTF-16BE 或 UTF-16LE）来存储字符串。
5. BLOB：
    - 代表“二进制大对象”，用于存储二进制数据，存储的内容完全按照输入的原样存储。

## 在Room中使用Date日期类型

```kotlin
data class CompletionRecord(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    @NotNull val goalId: String,
    // 这块需要存储记录是哪一天
    @NotNull val completionTime: Date = Date(System.currentTimeMillis()),
)
```

```kotlin title="其中Date并没有对应的sqlite类型，如果此时要存储需要进行类型转换，以下是类型转换代码"
import androidx.room.TypeConverter

class SqliteConverts {
    @TypeConverter
    fun dateToLong(date: java.sql.Date): Long {
        return date.time
    }

    @TypeConverter
    fun longToDate(time: Long): java.sql.Date {
        return java.sql.Date(time)
    }
}
```

```kotlin title="在数据库入口添加转换器代码"
@TypeConverters(SqliteConverts::class)
abstract class GoalDatabase : RoomDatabase()
```
