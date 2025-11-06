---
title: ViewModel
sidebar_label: ViewModel
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# ViewModel

ViewModel 是 Android Jetpack 库的一部分，旨在以声明式方式存储和管理 UI 相关的数据。ViewModel 的主要目的是解决设备配置更改（如屏幕旋转）时数据丢失的问题，并帮助组织代码以使数据管理与 UI 控制器（如活动和片段）解耦。以下是 ViewModel 的一些关键特点和用途：

## 生命周期意识

-   ViewModel 对象会自动管理其生命周期，以确保数据在配置更改时不会丢失。- ViewModel 存活在一个 Activity 或 Fragment 的生命周期之上，这意味着即使在这些组件因配置更改（如屏幕旋转）而被销毁和重建，ViewModel 中的数据仍然保持不变。

## 数据管理和解耦

-   ViewModel 使数据管理逻辑与 UI 控制器解耦，从而简化了架构。这有助于实现更清晰、更模块化的代码设计，使数据处理逻辑更容易测试和重用。
-   ViewModel 可以包含所有更新 UI 所需的信息和逻辑，包括进行网络请求、访问数据库和处理数据。

## 使用场景

-   保存和管理 UI 相关数据：ViewModel 存储用户界面所需的数据，例如用户输入、服务器响应或应用状态。
-   处理配置更改：在屏幕旋转等配置更改后，ViewModel 保证数据不丢失。
-   数据加载：ViewModel 可以与 LiveData 或 StateFlow 配合使用，以观察数据变化并在数据变化时更新 UI，支持构建响应式 UI。
-   处理业务逻辑：ViewModel 可以包含业务逻辑，从而使这些逻辑与 UI 控制器分离，提高代码的可测试性。

## 使用示例

创建 ViewModel 的基本步骤通常包括定义一个扩展 ViewModel 类的类，并在 UI 控制器（如 Activity 或 Fragment）中获取其实例：

```kt
// viewModel
import androidx.lifecycle.ViewModel
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue

class ExampleViewModel : ViewModel() {
    // 使用 mutableStateOf 来创建可观察的状态
    var counter by mutableStateOf(0)
        private set

    fun incrementCounter() {
        counter++
    }
}

// ui
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ExampleScreen() {
    // 在 Composable 函数中获取 ViewModel 实例
    val exampleViewModel: ExampleViewModel = viewModel()

    Button(onClick = { exampleViewModel.incrementCounter() }) {
        Text("Counter is ${exampleViewModel.counter}")
    }
}
```

## 注意事项

-   不要在 ViewModel 中持有 Context 的引用：为了避免内存泄露，ViewModel 不应该持有任何 Context 的引用。如果需要 Context（例如，访问资源），请使用 AndroidViewModel，它接受应用的 Context 作为参数。
-   不要在 ViewModel 中引用视图或 Activity 实例：这样做可能导致内存泄露或不一致的状态。
-   ViewModel 是现代 Android 应用开发的重要组件，有助于实现更健壮、可维护和测试的应用架构。
