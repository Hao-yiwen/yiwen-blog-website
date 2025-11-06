---
title: 常用的flutter组件
sidebar_label: 常用的flutter组件
date: 2024-09-13
last_update:
  date: 2024-09-13
---

# 常用的flutter组件

1. 基础组件（Basic Widgets）

这些是最基本的 Flutter 组件，用于构建简单的 UI。

-   Text：显示一行文本。
-   Container：一个通用的容器，可以包含其他子组件，并可以设置尺寸、边距、装饰等。
-   Column：垂直方向的布局，子组件按顺序从上到下排列。
-   Row：水平方向的布局，子组件从左到右排列。
-   Stack：允许子组件叠加，通常用于定位元素。
-   Padding：添加内边距的组件。
-   Align：将子组件对齐到父组件的特定位置。
-   Center：将子组件居中显示。
-   DecoratedBox：装饰子组件。

2. 输入与表单组件（Input & Forms Widgets）

这些组件用于构建用户输入表单。

-   TextField：用户输入文本的组件。
-   Checkbox：复选框，允许用户选择。
-   Radio：单选按钮，允许用户选择单个选项。
-   Switch：开关按钮，用户可以切换两个状态。
-   Slider：滑动条，用于选择值。
-   Form：表单容器，可以包含多个输入组件，并进行表单验证。
-   DropdownButton：下拉选择框。

3. 布局组件（Layout Widgets）

用于布局和排列子组件的容器。

-   Expanded：用于在 Row 或 Column 中扩展子组件，占据剩余的空间。
-   SizedBox：定义一个具有固定尺寸的盒子。
-   Wrap：可以让子组件在行中排列，行满后自动换行。
-   GridView：创建网格布局，子组件排列成行和列。
-   ListView：垂直滚动列表，显示一系列子组件。
-   Flex：在 Row 和 Column 之上，它允许在两个方向上灵活布局。
-   Table：表格布局，允许子组件排列成行和列。

4. 交互组件（Interactive Widgets）

这些组件支持用户交互，如按钮和手势检测。

-   ElevatedButton：一个凸起的按钮，带有阴影和背景颜色。
-   TextButton：一个纯文本按钮。
-   IconButton：显示图标的按钮。
-   FloatingActionButton：悬浮按钮，通常用于主要操作。
-   GestureDetector：手势检测组件，可以检测点击、拖动、双击等操作。
-   InkWell：带有水波纹效果的点击组件。
-   Draggable：可拖动的组件，支持拖放操作。

5. 导航与路由（Navigation & Routing）

这些组件用于处理屏幕导航和路由。

-   Navigator：管理页面堆栈的组件，用于处理导航。
-   PageView：页面视图，用于水平滑动切换页面。
-   Drawer：侧边抽屉菜单。
-   BottomNavigationBar：底部导航栏，用于在多个页面之间导航。
-   TabBar：选项卡导航栏。
-   TabBarView：配合 TabBar 显示内容的页面视图。

6. 动画与绘制组件（Animation & Painting Widgets）

这些组件用于处理动画和自定义绘制。

-   AnimatedContainer：容器的属性（如尺寸、颜色）可以在一段时间内变化。
-   AnimatedOpacity：用于透明度变化的动画。
-   FadeTransition：渐隐渐显动画。
-   ScaleTransition：缩放动画。
-   CustomPaint：用于自定义绘制的组件，允许你在画布上绘制内容。
-   Hero：两个页面间的元素动画，用于创建共享元素的过渡动画。

7. 滚动组件（Scrolling Widgets）

用于处理滚动的组件。

-   SingleChildScrollView：滚动视图，允许子组件在一个方向上滚动。
-   ListView：常用的列表视图，可以垂直滚动。
-   GridView：网格布局，可以滚动。
-   CustomScrollView：允许你组合 Sliver 组件创建复杂的滚动布局。
-   SliverList：配合 CustomScrollView 使用，创建一个惰性加载的滚动列表。
-   SliverGrid：用于网格布局的 Sliver 组件。

8. 对话框与弹出组件（Dialogs & Popups）

这些组件用于显示对话框、模态窗口和弹出菜单。

-   AlertDialog：警告对话框，用于向用户显示提示信息。
-   SimpleDialog：简单对话框，提供选项给用户选择。
-   BottomSheet：从屏幕底部弹出的面板。
-   SnackBar：显示短暂消息的提示条。
-   PopupMenuButton：弹出菜单按钮。

9. 状态管理组件（State Management Widgets）

这些组件用于处理和管理状态。

-   StatefulWidget：具有状态的组件，允许更新界面。
-   InheritedWidget：可以跨组件树共享数据的组件。
-   Provider：常用的第三方状态管理解决方案。

10. 图形与媒体组件（Graphics & Media Widgets）

这些组件用于显示图像、视频和其他媒体。

-   Image：显示图片，支持网络图片、本地图片等。
-   Icon：用于显示图标。
-   VideoPlayer：用于播放视频的组件。
-   ImageIcon：可以用图像作为图标。

11. 辅助与装饰组件（Helper & Decoration Widgets）

这些组件用于添加装饰或其他辅助功能。

-   ClipRRect：用于裁剪子组件，圆角矩形裁剪。
-   Opacity：设置子组件的透明度。
-   DecoratedBox：用于绘制装饰，背景颜色、边框等。
-   FittedBox：缩放子组件以适应父容器。

12. 平台特定组件（Platform Specific Widgets）

Flutter 支持在不同平台上使用原生 UI 元素。

-   CupertinoButton：iOS 风格的按钮。
-   CupertinoSlider：iOS 风格的滑动条。
-   CupertinoNavigationBar：iOS 风格的导航栏。
-   MaterialApp：用于 Material Design 风格的应用。
-   CupertinoApp：用于 Cupertino 风格的应用。

13. 国际化组件（Internationalization Widgets）

用于支持多语言和本地化。

-   Localizations：管理应用的本地化信息。
-   Locale：代表一个区域设置（语言和国家）。
-   Intl：用于国际化和多语言支持的第三方库。

14. 布局约束与尺寸控制组件（Constraints & Size Widgets）

控制布局中组件尺寸和位置的约束。

-   ConstrainedBox：施加大小限制的容器。
-   FractionallySizedBox：基于父组件的比例控制尺寸。
-   IntrinsicHeight：根据子组件的固有高度调整自身尺寸。
-   IntrinsicWidth：根据子组件的固有宽度调整自身尺寸。

15. 常用的sliver组件

:::info
sliver组件只支持线形布局，无法进行z轴布局。如果要进行z轴布局则需要在一个SliverToBoxAdapter完成布局操作。
:::

-   SliverAppBar：SliverAppBar 是 Flutter 中用于创建可以随着用户滚动进行折叠、展开或固定的应用栏。它经常用于创建类似于 CollapsingToolbarLayout 的效果。
-   SliverList：SliverList 是用于显示列表的 Sliver 组件，类似于 ListView，它按需加载列表项，适合惰性加载和长列表。
-   SliverGrid：SliverGrid 是一个用于显示网格布局的 Sliver 组件，类似于 GridView，但它与 CustomScrollView 一起使用时可以创建惰性加载的网格布局。
-   SliverToBoxAdapter：SliverToBoxAdapter 允许你将一个普通的 Widget 嵌入到 Sliver 结构中。如果你想在 CustomScrollView 中加入一个非 Sliver 的普通组件，就可以使用这个组件。
-   SliverPersistentHeader：SliverPersistentHeader 是一种特殊的 Sliver，用于创建在滚动时保持固定或浮动的头部。例如，创建类似于 TabBar 固定在顶部的效果。
-   SliverFillRemaining：SliverFillRemaining 是一个用于填充剩余空间的 Sliver 组件。它确保其子组件填满 CustomScrollView 中的剩余空间，通常用于在列表滚动结束时提供一些额外内容或填充空白。
-   SliverFixedExtentList：SliverFixedExtentList 是一种优化的 SliverList，用于显示具有固定高度的列表项。与 SliverList 不同，它的每一项都具有相同的高度，因此在布局时更加高效。
-   SliverOverlapAbsorber 和 SliverOverlapInjector：这两个组件通常配合 NestedScrollView 使用，用于处理嵌套滚动中的重叠区域。SliverOverlapAbsorber 吸收重叠的部分，而 SliverOverlapInjector 则将这些重叠注入回视图。
