# WorkManager

WorkManager提供了灵活的API来支持不同类型的后台任务调度需求，主要包括一次性任务（OneTimeWorkRequest）和周期性任务（PeriodicWorkRequest）。以下是这两种使用方式的详细介绍和示例。

## 一次性任务（OneTimeWorkRequest）

一次性任务是指那些只需执行一次的任务。例如，上传文件、同步一次数据等。你可以为这些任务设置约束（如网络类型），并且可以添加输入数据和接收输出数据。

示例：创建一个一次性任务来上传文件，仅在网络连接可用时执行。

```kt
// 定义Worker类
class UploadWorker(appContext: Context, workerParams: WorkerParameters):
        Worker(appContext, workerParams) {
    override fun doWork(): Result {
        // 执行上传操作
        uploadFile()
        // 返回任务执行结果
        return Result.success()
    }
}

// 创建并安排一次性任务
val uploadWorkRequest = OneTimeWorkRequestBuilder<UploadWorker>()
    .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
    .build()

WorkManager.getInstance(context).enqueue(uploadWorkRequest)
```

## 周期性任务（PeriodicWorkRequest）

周期性任务用于需要定期执行的任务，如每天或每隔几小时同步数据。注意，由于系统的限制，周期性任务的最小间隔时间是15分钟。

示例：创建一个周期性任务每天同步数据，但只在设备充电时执行。

```kt
// 定义Worker类
class SyncWorker(appContext: Context, workerParams: WorkerParameters):
        Worker(appContext, workerParams) {
    override fun doWork(): Result {
        // 执行同步操作
        syncData()
        // 返回任务执行结果
        return Result.success()
    }
}

// 创建并安排周期性任务
val syncWorkRequest = PeriodicWorkRequestBuilder<SyncWorker>(24, TimeUnit.HOURS) // 每24小时执行一次
    .setConstraints(Constraints.Builder().setRequiresCharging(true).build()) // 仅在充电时执行
    .build()

WorkManager.getInstance(context).enqueue(syncWorkRequest)
```

## 高级使用方式

除了基本的一次性和周期性任务，WorkManager还支持以下高级功能：

-   链式任务：可以将多个任务链接起来按顺序执行。
-   唯一工作序列：可以确保即使多次请求也只有一个任务或任务序列在执行。
-   输入/输出数据：可以为任务传递数据并从任务中获取结果。
-   灵活的失败重试策略：可以为任务设置重试策略。
    示例：链式执行两个任务，首先下载数据，然后处理数据。

```kt
val downloadRequest = OneTimeWorkRequestBuilder<DownloadWorker>().build()
val processDataRequest = OneTimeWorkRequestBuilder<ProcessDataWorker>().build()

WorkManager.getInstance(context)
    .beginWith(downloadRequest)
    .then(processDataRequest)
    .enqueue()
```
