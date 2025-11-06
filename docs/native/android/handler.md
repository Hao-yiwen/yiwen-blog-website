---
title: Handler
sidebar_label: Handler
date: 2024-07-02
last_update:
  date: 2024-07-02
---

# Handler

Handler 是 Android 中用于处理消息和执行任务的一个重要组件，通常与 Looper 和 MessageQueue 一起使用，以实现线程间通信和任务调度。以下是 Handler 的详细介绍：

## 主要功能

1. 处理消息 (Messages)
    - Handler 可以发送和处理消息，通过 sendMessage() 方法将消息放入线程的消息队列 (MessageQueue)，然后在 Handler 的 handleMessage() 方法中处理这些消息。
2. 执行任务 (Runnables)
    - Handler 还可以通过 post() 方法将 Runnable 对象放入消息队列中，以在特定的时间或延迟后在关联的线程上执行。

## 工作机制
-	Looper：每个线程可以有一个 Looper 对象，负责管理该线程的消息队列，并不断地从队列中取出消息进行处理。
-	MessageQueue：每个线程有一个消息队列，用于存储由 Handler 发送的消息和任务。
-	Handler：与一个 Looper 关联，负责发送消息和执行任务，以及在 handleMessage() 方法中处理消息。

## 示例代码
```java
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private TextView textView;
    private Handler handler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        textView = new TextView(this);
        setContentView(textView);

        handler = new Handler(Looper.getMainLooper()) {
            @Override
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                textView.setText("Message received: " + msg.what);
            }
        };

        // Sending a message to be handled
        handler.sendEmptyMessage(1);

        // Posting a Runnable to be executed
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                textView.setText("Runnable executed");
            }
        }, 2000);
    }
}
```