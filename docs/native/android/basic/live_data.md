# LiveData

liveData和stateFlow类似，是一个可观察的数据持有类，lieveData是生命周期感知的，只在活跃生命周期状态才会通知数据变化。

## 依赖

```gradle
dependencies {
    implementation "androidx.lifecycle:lifecycle-viewmodel-ktx:2.3.1"
    implementation "androidx.lifecycle:lifecycle-livedata-ktx:2.3.1"
}
```

## 创建 ViewModel

```kotlin
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class MainViewModel : ViewModel() {
    private val _textData = MutableLiveData<String>()
    val textData: LiveData<String> = _textData

    fun updateText(newText: String) {
        _textData.value = newText
    }
}
```

## 在 Activity 中观察 LiveData

```kotlin
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.activity.viewModels
import androidx.lifecycle.Observer
import com.example.myapp.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel.textData.observe(this, Observer { newText ->
            // 更新 UI
            binding.textView.text = newText
        })

        // 模拟数据更新
        binding.button.setOnClickListener {
            viewModel.updateText("Hello, LiveData!")
        }
    }
}
```

## liveData和stateflow比较（livedata能在java使用，但是stateflow不能在java使用）

### 相似点

-   观察者模式：LiveData 和 StateFlow 都基于观察者模式，允许组件订阅并响应数据变化。
-   UI 更新：它们都可以用于在数据变化时更新 UI，确保 UI 展示的数据是最新的。
-   生命周期感知：LiveData 是生命周期感知的，只在组件活跃时更新数据。而 StateFlow 本身不是生命周期感知的，但可以通过 Kotlin 的 Flow 在 - lifecycle-runtime-ktx 库的帮助下轻松实现生命周期感知。

### 不同点

1. 生命周期感知：

-   LiveData：自带生命周期感知能力，只在 LiveData 的观察者处于活跃状态时才会通知数据变化。
-   StateFlow：本身不是生命周期感知的，需要通过额外的代码（如使用 lifecycleScope 和 collect）来处理生命周期。

2. 默认值：

-   LiveData：不需要初始化时就有一个值。
-   StateFlow：需要在创建时就提供一个初始值。

3. 线程安全和并发：

-   LiveData：设计为主线程安全，通常在主线程上观察和修改。
-   StateFlow：是线程安全的，可以在任何线程上收集和发射数据。

4. 性能和功能：

-   LiveData：功能相对简单，主要支持基本的观察者模式。
-   StateFlow：作为 Kotlin Coroutines 的一部分，支持更复杂的流操作，如合并、过滤、转换数据流等。

5. 用例：

-   LiveData：适合简单的数据绑定和数据通知场景。
-   StateFlow：由于支持更广泛的响应式流操作，适合需要利用这些特性的复杂场景或更现代的 Kotlin 项目。
