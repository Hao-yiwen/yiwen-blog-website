# MVVM

MVVM（Model-View-ViewModel）模式的产生是为了更好地分离关注点和简化用户界面逻辑，而不仅仅是因为 Controller 逻辑过于复杂。MVVM 主要针对的是以下几个方面的改进：

1. 数据绑定(核心，少了mvc中的手动更新视图步骤)

MVVM 模式引入了双向数据绑定机制，使得视图和数据模型之间可以自动同步。这减少了手动更新视图的需求，从而简化了代码并减少了错误。

2. 视图与逻辑的分离

在 MVC 模式中，Controller 有时会变得复杂，因为它需要处理大量的视图逻辑。MVVM 模式通过引入 ViewModel，将视图逻辑从 Controller 中分离出来，使得视图逻辑集中在 ViewModel 中，而视图（View）只负责展示数据。

3. 可测试性

在 MVVM 模式中，ViewModel 是一个纯粹的逻辑层，与视图无关，这使得它更容易进行单元测试。相比之下，MVC 中的 Controller 往往直接操作视图，使得测试变得复杂。

4. 更清晰的职责分工

MVVM 模式明确了每个组件的职责：

-	Model：负责处理数据和业务逻辑。
-	View：负责展示数据和用户交互。
-	ViewModel：作为视图和模型之间的中介，处理视图逻辑和数据转换。

MVVM 示例

以下是一个简单的 MVVM 示例，用于展示如何实现一个用户登录功能。

```swift
// User.swift
struct User {
    var username: String
    var password: String
}
```

```swift
// UserViewModel.swift
import Foundation

class UserViewModel {
    private var user: User

    var username: String {
        get { return user.username }
        set { user.username = newValue }
    }

    var password: String {
        get { return user.password }
        set { user.password = newValue }
    }

    init(user: User) {
        self.user = user
    }

    func updateUsername(_ username: String) {
        self.username = username
    }

    func updatePassword(_ password: String) {
        self.password = password
    }
}
```

```swift
// ContentView.swift
import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: UserViewModel

    var body: some View {
        VStack {
            TextField("Username", text: $viewModel.username)
            SecureField("Password", text: $viewModel.password)
            Button("Login") {
                // Handle login action
                print("Logging in with \(viewModel.username) and \(viewModel.password)")
            }
        }
        .padding()
    }
}
```

```swift
// Main.swift
import SwiftUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            let user = User(username: "JohnDoe", password: "password123")
            let viewModel = UserViewModel(user: user)
            ContentView(viewModel: viewModel)
        }
    }
}
```
