---
title: Android 后台任务调度与 WorkManager 详解
sidebar_label: 后台任务与 WorkManager
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# Android 后台任务调度与 WorkManager 详解

在 Android 平台上，实现后台任务调度有多种机制，不同 API 对应不同的可靠性与触发时机诉求。现代 Android 开发中，官方首选 **Jetpack WorkManager** 处理可延迟但必须保证执行的后台任务。

## 一、WorkManager 核心用法

`WorkManager` 主要支持**一次性任务（`OneTimeWorkRequest`）**与**周期性任务（`PeriodicWorkRequest`）**，即使应用退出或设备重启也能保证任务最终执行。

### 1. 一次性任务（OneTimeWorkRequest）

适用于日志上传、单次数据同步等场景，可附加网络状态、电量等约束条件：

```kotlin
class UploadWorker(appContext: Context, workerParams: WorkerParameters) :
    Worker(appContext, workerParams) {
    override fun doWork(): Result {
        uploadFile()
        return Result.success()
    }
}

val uploadWorkRequest = OneTimeWorkRequestBuilder<UploadWorker>()
    .setConstraints(
        Constraints.Builder()
            .setRequiredNetworkType(NetworkType.CONNECTED)
            .build()
    )
    .build()

WorkManager.getInstance(context).enqueue(uploadWorkRequest)
```

### 2. 周期性任务（PeriodicWorkRequest）

适用于定期备份或轮询同步（受系统省电限制，最短周期为 **15 分钟**）：

```kotlin
class SyncWorker(appContext: Context, workerParams: WorkerParameters) :
    Worker(appContext, workerParams) {
    override fun doWork(): Result {
        syncData()
        return Result.success()
    }
}

val syncWorkRequest = PeriodicWorkRequestBuilder<SyncWorker>(24, TimeUnit.HOURS)
    .setConstraints(
        Constraints.Builder()
            .setRequiresCharging(true)
            .build()
    )
    .build()

WorkManager.getInstance(context).enqueue(syncWorkRequest)
```

### 3. 链式任务编排與高级特性

WorkManager 还支持任务链顺序编排、唯一任务序列（Unique Work）、输入输出数据传递与指数退避重试策略：

```kotlin
val downloadRequest = OneTimeWorkRequestBuilder<DownloadWorker>().build()
val processDataRequest = OneTimeWorkRequestBuilder<ProcessDataWorker>().build()

WorkManager.getInstance(context)
    .beginWith(downloadRequest)
    .then(processDataRequest)
    .enqueue()
```

---

## 二、Android 后台调度方案横向对比

| 方案 | 适用场景 | 备注 |
| :--- | :--- | :--- |
| **WorkManager** | 需要可靠执行的可延迟后台任务（上传、同步） | Jetpack 官方推荐，底层自动适配 JobScheduler |
| **AlarmManager** | 必须在精确时间点唤醒触发的定时提醒（闹钟、日历） | 不适合依赖网络/充电条件的常规同步任务 |
| **JobScheduler** | 系统级条件触发任务（API 21+） | WorkManager 在现代系统上的底层实现机制 |
| **Foreground Service** | 用户可感知、需立即持续运行的任务（音乐播放、导航、下载） | 必须常驻通知栏提示 |

### 附：`AlarmManager` 与原生 `JobScheduler` 示例

```java title="AlarmManager 精确时间触发"
AlarmManager alarmManager = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
Intent intent = new Intent(context, AlarmReceiver.class);
PendingIntent pendingIntent = PendingIntent.getBroadcast(context, 0, intent, PendingIntent.FLAG_IMMUTABLE);

alarmManager.set(AlarmManager.RTC_WAKEUP, triggerTimeMillis, pendingIntent);
```

```java title="原生 JobScheduler 条件调度"
ComponentName componentName = new ComponentName(context, MyJobService.class);
JobInfo jobInfo = new JobInfo.Builder(JOB_ID, componentName)
        .setRequiredNetworkType(JobInfo.NETWORK_TYPE_UNMETERED)
        .setRequiresCharging(true)
        .build();

JobScheduler jobScheduler = (JobScheduler) context.getSystemService(Context.JOB_SCHEDULER_SERVICE);
jobScheduler.schedule(jobInfo);
```
