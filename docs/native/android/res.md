# res资源优先级

## 使用sourceSets有多个res目录的时候

```xml
android {
    ...
    sourceSets {
        main {
            res.srcDirs = [
                'src/main/res/common',
                'src/main/res/home',
                'src/main/res/video',
                'src/main/res/micro',
                'src/main/res/mine'
            ]
        }
    }
}
```

如果 `src/main/res/common` 和 `src/main/res/home` 目录中都有一个名为 strings.xml 的文件：

-   `src/main/res/common/values/strings.xml`
-   `src/main/res/home/values/strings.xml`

在编译时，`src/main/res/common/values/strings.xml` 文件中的内容将覆盖 `src/main/res/home/values/strings.xml` 中的内容，因为 `src/main/res/common` 在 `res.srcDirs` 配置中排在第一位，具有最高优先级。

## 主应用（app module）和其他模块（library modules）都有 res 目录

优先级通常遵循以下规则：

1. 主应用模块的资源覆盖一切：

-   如果主应用模块和库模块都有相同名称的资源文件，主应用模块中的资源文件会覆盖库模块中的资源文件。

2. 库模块之间的优先级：

-   当多个库模块之间存在相同的资源文件时，优先级取决于它们在依赖关系中的位置。直接依赖的库模块会覆盖间接依赖的库模块中的资源文件。

3. 依赖库的资源优先级最低：

-   最后，依赖库中的资源文件会被主应用模块和库模块中的资源文件覆盖。
