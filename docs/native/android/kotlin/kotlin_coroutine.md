# kotlin中的协成并发请求

有一个典型示例是在`viewModel`中同时进行数据库读写和进行网络请求。

```kt
viewModelScope.launch {
    _status.postValue("Loading...")

    // 并行执行网络请求和数据库写入
    coroutineScope {
        // 确保所有子协程完成
        val networkJob = async(Dispatchers.IO) {
            try {
                val response = apiService.fetchData()
                if (response.isSuccessful) {
                    response.body()?.let { myDao.insertData(it) }
                    "Data fetched and stored successfully"
                } else {
                    "Failed to fetch data: ${response.errorBody()?.string()}"
                }
            } catch (e: Exception) {
                "Network request failed: ${e.message}"
            }
        }
        // 网络请求
        _status.postValue(networkJob.await()) // You can modify this based on your logic
    }
}
```