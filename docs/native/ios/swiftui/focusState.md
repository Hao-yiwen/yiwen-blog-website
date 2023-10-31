---
sidebar_position: 3
---

# @FocusState说明

`@FocusState`的初始化和`@state`不太一样

## @state

```swift
struct ContentView: View {
    @State var idString: String
}

ContentView(idString: "123")
```

## @FocusState

```swift
@FocusState var isFouced: Bool

VStack {
    // ...省略
}
.onAppear {
    DispatchQueue.global(qos: .default).async {
        isFouced = true
    }
//            isFouced = true
}
```

## 总结

`@FocusState`需要只能在页面出现后初始化,不像`@state`可以直接传值进去初始化。
