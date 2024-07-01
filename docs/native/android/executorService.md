# ExecutorService

ExecutorService 是 Java 提供的一个框架，用于简化并发编程中的线程管理和任务调度。它位于 java.util.concurrent 包中，提供了多种方法来管理线程和任务，确保更高效和更可靠的并发编程。以下是 ExecutorService 的详细介绍：

主要功能

1. 线程池管理：
    - ExecutorService 提供了线程池，可以管理和重用多个线程，避免频繁创建和销毁线程带来的开销。
    - 常见的线程池实现包括 ThreadPoolExecutor、ScheduledThreadPoolExecutor 等。
2. 任务调度：
    - 可以使用 ExecutorService 提交任务（实现 Runnable 或 Callable 接口），线程池会自动调度这些任务。
    - 提供了 submit 方法，返回 Future 对象，通过 Future 可以获取任务的执行结果或取消任务。
3. 定时任务：
    - ScheduledExecutorService 接口扩展了 ExecutorService，提供了调度任务在特定时间执行或周期性执行的功能。

## 使用

```java
//提交任务
ExecutorService executorService = Executors.newFixedThreadPool(10);
executorService.submit(new RunnableTask());

//获取结果
Future<String> future = executorService.submit(new CallableTask());
String result = future.get();  // 获取任务执行结果

//定时任务
ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(5);
scheduledExecutorService.schedule(new RunnableTask(), 5, TimeUnit.SECONDS);

//关闭服务
executorService.shutdown();  // 平滑关闭，等待所有任务完成
executorService.shutdownNow();  // 立即关闭，尝试取消正在执行的任务
```

## 示例

```java
import java.util.concurrent.*;

public class ExecutorServiceExample {
    public static void main(String[] args) {
        ExecutorService executorService = Executors.newFixedThreadPool(3);

        // 提交 Runnable 任务
        executorService.execute(new RunnableTask());

        // 提交 Callable 任务并获取结果
        Future<String> future = executorService.submit(new CallableTask());
        try {
            String result = future.get();
            System.out.println("Callable Task Result: " + result);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // 关闭 ExecutorService
        executorService.shutdown();
    }
}

class RunnableTask implements Runnable {
    @Override
    public void run() {
        System.out.println("Runnable Task Executed");
    }
}

class CallableTask implements Callable<String> {
    @Override
    public String call() {
        return "Callable Task Executed";
    }
}
```

## ExecutorService和Thread区别

1. 线程管理

Thread需要自己管理，ExecutorService通过线程池管理。

2. 线程池复用

-   Thread每次都创建一个新的线程，线程使用完毕后会被GC回收，频繁创建和销毁线程会带来额外的开销。
-   使用ExecutorService，线程池中的线程可以被复用，任务执行完毕后，线程不会被销毁，而是返回线程池中待命，可以继续执行新的任务。
