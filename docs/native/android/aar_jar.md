# aar和jar区别

AAR (Android Archive) 和 JAR (Java Archive) 文件格式都用于打包和分发代码，但它们服务于不同的目的，尤其是在 Android 应用开发中。

## JAR 文件

-   定义：JAR 文件是一种基于 ZIP 的归档格式，用于分发和打包 Java 类文件、相关元数据和资源。它们可以包含任何类型的文件，但通常不包括 Android 特有的资源和元数据。
-   用途：JAR 文件通常用于分发 Java 库。在 Android 开发中，它们可以用于包含纯 Java 代码，但不能包含 Android 资源（如布局文件、图片、字符串资源等）。
-   兼容性：JAR 文件可以在任何 Java 平台上使用，不仅限于 Android。

## AAR 文件

-   定义：AAR 文件是专门为 Android 开发设计的归档格式。它扩展了 JAR 文件，允许包含 Android 特有的资源和元数据，例如布局文件（XML）、图片资源（PNG、JPG 等）、编译后的资源（R.class）和 AndroidManifest.xml 文件等。
-   用途：AAR 文件用于分发 Android 库。它们允许开发者创建和分享包含布局、图片、样式等 Android 资源的库，这在创建可重用的 UI 组件和模块时非常有用。
-   兼容性：AAR 文件专为 Android 应用开发设计，因此只能在 Android 项目中使用。

## 主要区别

-   资源支持：最主要的区别是 AAR 支持 Android 特定资源和元数据，而 JAR 不支持。
-   用途：JAR 主要用于 Java 应用或库的分发，适用于所有 Java 平台。AAR 专为 Android 库设计，支持 Android 特有的功能和资源。
-   文件结构：AAR 文件包含额外的文件夹和文件（如 res/、assets/、AndroidManifest.xml 等），这些在 JAR 文件中通常不存在。

## 选择哪一个

-   如果你需要创建一个纯 Java 库，或者需要在非 Android 环境中重用代码，JAR 文件可能是更好的选择。
-   如果你在开发 Android 库，并需要包含 Android 资源和元数据，那么 AAR 文件是必需的。
-   总的来说，选择 JAR 或 AAR 格式取决于你的特定需求，以及你的代码和资源是否专门用于 Android 平台。
