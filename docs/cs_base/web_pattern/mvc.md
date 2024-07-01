# MVC

下面的示例存在问题，view的更新和model操作都在controller层。但是实际上的mvc并不是这样的。实际工作过程：

view接受io事件，然后io事件传递给controller层。controller层更新model数据，然后view监听model数据，从而更新视图。所以view和model是耦合的没有完全解耦。而下面的例子更相当于mvp模式。通过主持人层来控制和moedl和view，所有操作都在主持人层。做到了完全解耦。

## 文档

[mvc](https://www.ruanyifeng.com/blog/2015/02/mvcmvp_mvvm.html)

~~MVC（Model-View-Controller）模式属于行为型模式，虽然它本身不完全是经典的设计模式中的一种，但它主要处理对象和类之间的职责分配和交互方式，因此归类为行为型模式更为恰当。~~

## MVC 模式简介

-   ~~Model（模型）：代表应用程序的数据和业务逻辑。模型直接管理数据、逻辑和规则。~~
-   ~~View（视图）：显示数据的可视化部分。视图从模型中获取数据并显示，但不处理数据的逻辑。~~
-   ~~Controller（控制器）：接受用户输入并调用模型和视图完成用户的需求。控制器通过更新模型来影响视图。~~

## 行为型模式

~~行为型模式关注对象和类之间的职责分配和通信方式，包括算法、职责链、命令、迭代器、中介者、备忘录、观察者、状态、策略、模板方法和访问者等模式。行为型模式的主要目标是简化对象之间的通信和控制流程，促进系统内的对象协作。~~

## MVC 示例

以下是一个简单的 MVC 示例，展示了如何实现一个用户登录功能。

```java
// User.java
public class User {
    private String username;
    private String password;

    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }

    // Getters and setters
}
```

```java
// LoginView.java
public class LoginView {
    public void printUserDetails(String username, String password){
        System.out.println("User: ");
        System.out.println("Username: " + username);
        System.out.println("Password: " + password);
    }
}
```

```java
// UserController.java
public class UserController {
    private User model;
    private LoginView view;

    public UserController(User model, LoginView view){
        this.model = model;
        this.view = view;
    }

    public void setUserName(String username){
        model.setUsername(username);
    }

    public String getUserName(){
        return model.getUsername();
    }

    public void setUserPassword(String password){
        model.setPassword(password);
    }

    public String getUserPassword(){
        return model.getPassword();
    }

    public void updateView(){
        view.printUserDetails(model.getUsername(), model.getPassword());
    }
}
```
