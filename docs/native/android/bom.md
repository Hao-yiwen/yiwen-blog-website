# Gradle中的Bom依赖管理

## 介绍

Bill of Materials（BOM）依赖管理是一种在多个项目或模块中统一管理软件库版本的策略。这种方法特别用于 Maven 和 Gradle 这样的构建工具，帮助简化和标准化依赖项版本的管理。通过使用 BOM，开发团队可以确保在不同的项目中使用相同的库版本，从而减少兼容性问题，并提高项目的维护效率。

## 实现

### Maven

1. 创建一个新的 Maven 项目：这个项目不包含实际的业务逻辑代码，仅用于定义依赖项和版本。

2. 编辑 pom.xml：将该项目的 packaging 类型设置为 pom。然后，在 `<dependencyManagement>` 部分定义你想要管理的依赖项及其版本。例如：

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>example-bom</artifactId>
    <version>1.0.0</version>
    <packaging>pom</packaging>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-starter-web</artifactId>
                <version>2.3.1.RELEASE</version>
            </dependency>
            <!-- 更多依赖项 -->
        </dependencies>
    </dependencyManagement>
</project>
```

3. 发布 BOM：完成 BOM 的定义后，将其发布到 Maven 仓库，以便在其他项目中引用。

### gradle

在 Gradle 中，创建和使用 BOM 的方式略有不同，因为 Gradle 原生支持 BOM 的概念是从 5.0 版本开始的。

1. 定义 BOM：在任何 Gradle 项目中，你可以定义一个 BOM，但通常不需要像 Maven 那样创建一个单独的项目。相反，你可以在项目的 build.gradle 文件中定义依赖项，并使用 java-platform 插件来声明这是一个 BOM 项目。例如：

```kt
plugins {
    id 'java-platform'
}

javaPlatform {
    allowDependencies()
}

dependencies {
    constraints {
        api 'org.springframework.boot:spring-boot-starter-web:2.3.1.RELEASE'
        // 更多依赖项
    }
}
```

## 使用

### maven

1. 添加 BOM 到 `<dependencyManagement>` 部分：

在项目的 pom.xml 文件中，你可以通过添加一个 `<dependencyManagement>` 部分并在其中引入 BOM 作为一个依赖项来使用 BOM。例如，如果你想使用 Spring Boot 的 BOM 来管理你的 Spring Boot 依赖项版本，可以这样做：

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>2.3.1.RELEASE</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

:::tip
`<scope>import</scope>` 的作用：
当你在 `<dependencyManagement>` 部分的一个依赖配置中使用 `<scope>import</scope>` 时，Maven 不会将这个依赖项作为项目的直接依赖来处理。相反，它会将指定的 BOM 文件中列出的依赖项及其版本号导入到当前项目的依赖管理上下文中。这意味着你可以在项目中直接使用 BOM 文件中定义的依赖项和版本，而不需要在每个依赖声明中重复这些版本号。

这种机制主要用于集中管理和维护项目依赖的版本号，特别是当你的项目依赖了许多相互关联的库时（如 Spring Boot 或其他大型框架）。通过这种方式，可以确保所有相关依赖项的版本号都是协调一致的，从而减少版本冲突的风险。
:::

2. 添加依赖项而不指定版本：

一旦你在 `<dependencyManagement>` 中引入了 BOM，就可以在 `<dependencies>` 部分添加你需要的依赖项而不用指定版本号，因为 BOM 已经为你定义了这些依赖项的版本：

```kt
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!-- 更多依赖项 -->
</dependencies>
```

### gradle

1. 引入 BOM：

从 Gradle 5.0 开始，你可以使用 platform 关键字和 enforcedPlatform 来引入 BOM。这告诉 Gradle 使用 BOM 中指定的版本作为依赖项的首选版本。例如，使用 Spring Boot 的 BOM：

```groxy
dependencies {
    implementation platform('org.springframework.boot:spring-boot-dependencies:2.3.1.RELEASE')

    // 通过 BOM 管理版本的依赖项
    implementation 'org.springframework.boot:spring-boot-starter-web'
    // 更多依赖项
}
```

2. 添加依赖项而不指定版本：

和 Maven 类似，一旦引入了 BOM，就可以添加依赖项而不需要显式指定版本号，Gradle 会根据 BOM 自动解析和使用正确的版本：

```kt
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    // 更多依赖项
}
```
