# Navigation Graph

## 介绍

Navigation Graph 是 Android Navigation 组件的核心部分，用于可视化地定义应用中所有的导航路径。这个图定义了应用中用户可以从一个界面（通常是 Fragment）导航到另一个界面的所有可能方式。Navigation Graph 是通过 XML 文件进行定义的，它描述了应用的导航结构，包括所有目的地（destination）和可能的动作（action）。

## codelab

[android-navigation](https://developer.android.com/codelabs/android-navigation#6)

## 使用

### 依赖

```gradle
//Navigation
implementation "androidx.navigation:navigation-fragment-ktx:$rootProject.navigationVersion"
implementation "androidx.navigation:navigation-ui-ktx:$rootProject.navigationVersion"
```

### 使用

```xml title="navigation grph"
<?xml version="1.0" encoding="utf-8"?>
<navigation xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/nav_graph"
    app:startDestination="@id/homeFragment">

    <fragment
        android:id="@+id/homeFragment"
        android:name="com.yiwen.java_view_other.fragemnt.HomeFragment"
        android:label="fragment_home"
        tools:layout="@layout/fragment_home" >
        <action
            android:id="@+id/next_action_1"
            app:destination="@id/detailFragment" />
    </fragment>
    <fragment
        android:id="@+id/detailFragment"
        android:name="com.yiwen.java_view_other.fragemnt.DetailFragment"
        android:label="fragment_detail"
        tools:layout="@layout/fragment_detail" />
</navigation>
```

```xml title="navigation activity"
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:id="@+id/main"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".NavigationFragmentActivity">

    <fragment
        android:id="@+id/fragment_host_navigation"
        android:name="androidx.navigation.fragment.NavHostFragment"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintTop_toTopOf="parent"
        app:navGraph="@navigation/nav_graph_test" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

```java title="navigation activity"
public class NavigationFragmentActivity extends AppCompatActivity {
    ActivityNavigationFragmentBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityNavigationFragmentBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.fragment_host_navigation);

        NavController navController = navHostFragment.getNavController();

    }
}
```

```java title="home fragment"
public class HomeFragment extends Fragment {
    public HomeFragment() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_home, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // 再次获取按钮
        Button button = view.findViewById(R.id.next_action);
        if (button != null) {
            button.setOnClickListener(v -> {
                //跳转到下一页
                Navigation.findNavController(v).navigate(R.id.detailFragment, null);
            });
        }
    }
}
```

## Navigation Graph 的功能和好处

1. 可视化管理导航流：通过一个集中的 XML 文件，你可以轻松查看和管理所有导航相关的路径。
2. 减少代码复杂性：通过使用 declarative XML 文件，减少了在代码中手动管理 fragment transactions 和 back stack 的需要。
3. 安全地传递数据：支持在导航时安全地传递数据，通过定义参数并为这些参数设置类型和默认值，降低了运行时错误。
4. Deep Linking 的集成：Navigation 组件支持直接在 Navigation Graph 中定义和处理 Deep Links，使得从网页或其他应用进入你的应用的特定页面变得容易。
5. 动画和过渡：支持在 Fragment 间切换时定义动画和过渡效果。

## Navigation Graph 在 Jetpack Compose 中的使用

与基于 Fragment 的传统 Android 应用不同，Jetpack Compose 采用了不同的方法来处理导航。Compose 是一种全新的 UI 工具包，用于构建原生的、响应式的 UI。在使用 Jetpack Compose 开发应用时，通常不会使用传统的 Navigation Graph XML 文件。而是使用 Compose 版本的 Navigation 组件，通常被称为 Compose Navigation。

## 总结

虽然在基于 Fragment 的传统 Android 应用开发中 Navigation Graph 是一个非常有用的工具，但在使用 Jetpack Compose 构建应用时，通常会采用 Compose 特有的导航方式。这种方式更加符合 Compose 的整体设计哲学，使得代码更加简洁和一致。
