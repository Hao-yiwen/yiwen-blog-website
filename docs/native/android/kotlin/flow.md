---
title: Flow和StateFlow
sidebar_label: Flow和StateFlow
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Flow和StateFlow

Kotlin Flows 是 Kotlin 为处理异步数据流（asynchronous stream）引入的一套 API。这些 API 是协程（coroutines）的一部分，专门用于处理时间上分散的一系列数据。Flows 允许你以非阻塞的方式工作，使得在多个线程之间传递数据变得简单且安全。

## 核心特性

-   非阻塞：Flows 构建在 Kotlin 协程之上，提供了一种非阻塞的方式来处理异步数据流。
-   冷流：Flows 是按需生成的（即仅当订阅者开始收集时，数据才开始流动），这与热流（始终处于活动状态并广播数据给所有订阅者）形成对比。
-   组合操作符：提供了丰富的操作符，如 map、filter、take 等，允许对数据流进行变换和组合。
-   背压管理：虽然 Flows 本身不直接处理背压问题（当生产数据的速率超过消费者处理速率时发生的问题），但是它可以通过协程的挂起来间接实现背压管理。

## 使用场景

-   网络请求：使用 Flows 处理异步的网络请求响应。
-   数据库操作：监听数据库变化并响应。
-   实时数据更新：如股票价格、游戏得分等实时数据的更新。
-   用户界面事件处理：监听和响应用户界面事件，如点击、滑动等。

## 示例代码

```kt
fun main() {
    runBlocking {
        launch {
            for (k in 1..3) {
                println("I'm not blocked $k")
                delay(100)
            }
        }
        simple().collect { value -> println(value) }
    }
}

fun simple(): Flow<Int> = flow{
    for (k in 1..3) {
        delay(100)
        emit(k)
    }
}
```

## Room使用collect来消费flow，然后转换为stateflow(热流)

### 使用collect来消费
```kt
// repogistory中的定义，room支持flow消费
fun searchAllAirport(): Flow<List<Airport>>


// viewModel中的消费
/**
* @description 解决数据库初始化问题
* - 在3月份使用数据库的时候发现，初次进入页面无法初始化数据库，经过debugger发现，问题是之前的airportRegistory方法是Flow类型，而Flow类型是冷流，
* 只有订阅了才会执行，所以在初次进入页面的时候，没有订阅，导致数据库没有初始化，所以现在我改成Suspend,问题解决了
* - 还有一个问题就是uistate的问题，uistate需要委托创建而不是初始化，例如
* val uiState by viewModel.uiState.collectAsState()
* @date: 2024/4/5 01:00 AM
*/
viewModelScope.launch(Dispatchers.IO) {
    var allAirport: List<Airport> = emptyList()
    airportRepository.searchAllAirport().collect { value ->
        {
            allAirport = value
        }
    }
    _uiState.update {
        _uiState.value.copy(
            allAirport = allAirport
        )
    }
}
```

### 使用standIn来消费，直接转换

```kt
val uiState: StateFlow<BusScheduleUiState> = offlineBusScheduleRepository.getSchedule().map {
    BusScheduleUiState(it)
}
    .stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(TIMEOUT_MILLIS),
        initialValue = BusScheduleUiState()
    )
```

stateIn会新建一个BusScheduleUiState 然后将flow转换成stateflow后再赋值。

#### stateIn前面的map作用

看下面示例：

```kt
data class UIState(val count: Int, val errorMessage: String? = null)

val countFlow: Flow<Int> = flowOf(1, 2, 3) // 仅作为示例
val uiStateFlow: Flow<UIState> = countFlow.map { count ->
    UIState(count = count)
}
```
如果原始数据是`Flow<Int>`类型，而你想要的结果是包含这些数据的UIState对象的流，那么通过转换，你会得到`Flow<UIState>`。
