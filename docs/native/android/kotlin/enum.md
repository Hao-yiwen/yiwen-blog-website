# kotlin中的枚举

在 Kotlin 中，枚举（enum）类是一种特殊的类类型，用于定义一组命名的常量。这些常量（或称为枚举项）是枚举类的实例，可以包含属性和方法。Kotlin 的枚举提供了一种类型安全的方式来表示一组固定的常量值，这可以使代码更加清晰和易于管理。以下是关于 Kotlin 中枚举的一些关键特性：

## 基本定义

枚举类通过 enum 关键字定义。例如，定义一个表示方向的枚举：

```kt
enum class Direction {
    NORTH, EAST, SOUTH, WEST
}
```

## 枚举项

每个枚举项本身就是枚举类的一个实例，它们是静态的、唯一的。枚举项之间用逗号分隔。

## 初始化

枚举类可以有初始化参数，这允许你为每个枚举项提供额外的信息或功能：

```kt
enum class Color(val rgb: Int) {
    RED(0xFF0000),
    GREEN(0x00FF00),
    BLUE(0x0000FF)
}
```

## 枚举属性和方法

枚举类可以包含属性和方法。例如，你可以添加一个方法来判断某个颜色是否为暖色：

```kt
enum class Color(val rgb: Int) {
    RED(0xFF0000),
    GREEN(0x00FF00),
    BLUE(0x0000FF);

    fun isWarm(): Boolean {
        return this == RED // 如果当前枚举项是 RED，返回 true，表示它是一个暖色
    }
}
fun main() {
    val redColor = Color.RED
    val blueColor = Color.BLUE

    println("Is RED warm? ${redColor.isWarm()}") // 预期输出：Is RED warm? true
    println("Is BLUE warm? ${blueColor.isWarm()}") // 预期输出：Is BLUE warm? false
}
```

### 特殊方法

-   values(): 返回一个数组，包含枚举类中定义的所有枚举项。
-   valueOf(value: String): 返回具有指定名称的枚举项。如果不存在该名称的枚举项，则抛出 IllegalArgumentException 异常。
-   name: 这是一个内建属性，返回枚举项的名称。
-   ordinal: 这是一个内建属性，返回枚举项在枚举类中的位置。

### Examples
#### 访问枚举实例
你可以直接通过枚举名访问其实例，例如
```kt
val currentMenu = MenuList.START
```
#### 使用枚举实例
```kt
when(currentMenu) {
    MenuList.START -> println("Selected Start Menu")
    MenuList.ENTREE -> println("Selected Entree")
    // 其他枚举项类似处理
}
```
#### 获取枚举的名字和位置
```kt
println("Menu name: ${currentMenu.name}") // 输出枚举项的名称，例如 "START"
println("Menu position: ${currentMenu.ordinal}") // 输出枚举项的位置，例如 0
```
#### 枚举遍历
```kt
for (menu in MenuList.values()) {
    println(menu.name)
}
```
#### 与字符串比较
```kt
val menuName = "START"
val menu = MenuList.valueOf(menuName)
```
