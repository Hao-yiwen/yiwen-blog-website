---
sidebar_position: 6
---

# 匿名内部类的一点说明

在`web`开发中通常使用回调函数来处理异步事件或者用户交互，而在`android`中也差不多类似，但是有区别的是`Java`中通常使用匿名内部类来处理这类事件。将处理事件进行简化，工作原理基本如下。

```java
// 假设有一个名为 Button 的类，它有一个方法 setOnClickListener，这个方法接受一个实现了 OnClickListener 接口的对象作为参数。OnClickListener 接口定义了一个方法 onClick，当按钮被点击时应该执行这个方法。

// Button
public class Button {
    private OnClickListener listener;

    public void setOnClickListener(OnClickListener listener) {
        this.listener = listener;
    }

    // 某个时刻，按钮被点击，触发了这个方法
    public void click() {
        if (listener != null) {
            listener.onClick(this);
        }
    }
}

public interface OnClickListener {
    void onClick(Button button);
}

// 位button设置点击事件的监听器
Button myButton = new Button();
myButton.setOnClickListener(new OnClickListener() {
    @Override
    public void onClick(Button button) {
        // 处理按钮点击
    }
});
// 使用Lambda简化
myButton.setOnClickListener(b -> {
    // 处理按钮点击
})

```