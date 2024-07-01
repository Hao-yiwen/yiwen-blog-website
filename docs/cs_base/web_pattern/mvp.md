# MVP

MVP（Model-View-Presenter）是一种软件架构模式，用于分离应用程序的逻辑部分（Model）和用户界面（View），并通过主持人（Presenter）进行连接。这种模式的目的是提高代码的可维护性、可测试性和灵活性。MVP模式在Android开发中非常流行，因为它能很好地应对Android平台的复杂性和多样性。

## 文档

[mvp](https://www.ruanyifeng.com/blog/2015/02/mvcmvp_mvvm.html)

## 代码

```java
public class User {
    private String firstName;
    private String lastName;

    public User(String firstName, String lastName) {
        this.firstName = firstName;
        this.lastName = lastName;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }
}

public class UserPresenter {
    private User user;
    private UserView view;

    public UserPresenter(UserView view) {
        this.view = view;
        this.user = new User("John", "Doe");
    }

    public void loadUserData() {
        view.showUserData(user.getFirstName(), user.getLastName());
    }

    public void saveUserData(String firstName, String lastName) {
        user.setFirstName(firstName);
        user.setLastName(lastName);
        view.showUserData(user.getFirstName(), user.getLastName());
    }
}

public interface UserView {
    void showUserData(String firstName, String lastName);
}

public class MainActivity extends AppCompatActivity implements UserView {

    private EditText firstNameEditText;
    private EditText lastNameEditText;
    private TextView displayTextView;
    private Button saveButton;
    private UserPresenter presenter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化视图
        firstNameEditText = findViewById(R.id.firstNameEditText);
        lastNameEditText = findViewById(R.id.lastNameEditText);
        displayTextView = findViewById(R.id.displayTextView);
        saveButton = findViewById(R.id.saveButton);

        // 初始化Presenter
        presenter = new UserPresenter(this);
        presenter.loadUserData();

        // 设置保存按钮的点击监听器
        saveButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                presenter.saveUserData(
                        firstNameEditText.getText().toString(),
                        lastNameEditText.getText().toString()
                );
            }
        });
    }

    @Override
    public void showUserData(String firstName, String lastName) {
        String displayText = "First Name: " + firstName + "\nLast Name: " + lastName;
        displayTextView.setText(displayText);
    }
}
```

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <EditText
        android:id="@+id/firstNameEditText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="First Name" />

    <EditText
        android:id="@+id/lastNameEditText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:hint="Last Name" />

    <Button
        android:id="@+id/saveButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="Save" />

    <TextView
        android:id="@+id/displayTextView"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:paddingTop="16dp" />
</LinearLayout>
```