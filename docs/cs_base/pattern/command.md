# 命令行模式

命令行模式（Command Pattern）是一种行为型设计模式，它将一个请求封装成一个对象，从而使你可以用不同的请求对客户进行参数化；对请求排队或记录请求日志，以及支持可撤销的操作。命令模式的核心是创建命令对象，这个对象封装了执行操作所需的所有信息。

## 命令行模式的主要角色

1.	Command：命令接口，声明执行操作的方法。
2.	ConcreteCommand：具体命令类，实现了Command接口，负责调用接收者的相关操作。
3.	Receiver：接收者，具体执行命令相关操作的类。
4.	Invoker：调用者，负责调用命令对象执行请求。
5.	Client：客户端，创建具体的命令对象并设置其接收者。

## 命令行模式的实现

举个例子，假设我们有一个电灯（Light）类，客户端可以通过命令对象来打开或关闭电灯。

```java
// Command接口
public interface Command {
    void execute();
}


// 具体命令类，用于打开电灯
public class LightOnCommand implements Command {
    private Light light;

    public LightOnCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.on();
    }
}

// 具体命令类，用于关闭电灯
public class LightOffCommand implements Command {
    private Light light;

    public LightOffCommand(Light light) {
        this.light = light;
    }

    @Override
    public void execute() {
        light.off();
    }
}

// 接收者类
public class Light {
    public void on() {
        System.out.println("The light is on");
    }

    public void off() {
        System.out.println("The light is off");
    }
}

// 调用者类
public class RemoteControl {
    private Command command;

    public void setCommand(Command command) {
        this.command = command;
    }

    public void pressButton() {
        command.execute();
    }
}

public class Client {
    public static void main(String[] args) {
        // 创建接收者对象
        Light light = new Light();

        // 创建具体命令对象，并设置其接收者
        Command lightOn = new LightOnCommand(light);
        Command lightOff = new LightOffCommand(light);

        // 创建调用者，并设置命令
        RemoteControl remote = new RemoteControl();

        // 开灯
        remote.setCommand(lightOn);
        remote.pressButton();

        // 关灯
        remote.setCommand(lightOff);
        remote.pressButton();
    }
}
```