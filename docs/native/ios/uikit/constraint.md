# UIkit中的约束

:::info
以下代码都是自动布局(autolayout)
:::

1. 尺寸约束（Size Constraints）

    - 固定尺寸：设置视图的固定宽度或高度。

    ```swift
    let view = UIView()
    view.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
        view.widthAnchor.constraint(equalToConstant: 100),
        view.heightAnchor.constraint(equalToConstant: 50)
    ])
    ```

    - 比例尺寸：设置视图的宽度或高度与另一个视图的宽度或高度的比例关系。

2. 位置约束（Position Constraints）

    - 边距约束：设置视图的顶部、底部、左侧或右侧与父视图或其他视图的距离。

    ```swift
    let parentView = UIView()
    let childView = UIView()
    parentView.addSubview(childView)
    childView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
        childView.leadingAnchor.constraint(equalTo: parentView.leadingAnchor, constant: 20),
        childView.trailingAnchor.constraint(equalTo: parentView.trailingAnchor, constant: -20),
        childView.topAnchor.constraint(equalTo: parentView.topAnchor, constant: 10),
        childView.bottomAnchor.constraint(equalTo: parentView.bottomAnchor, constant: -10)
    ])
    ```

    - 居中约束：将视图在父视图或另一个视图中水平或垂直居中。

    ```swift
    let parentView = UIView()
    let childView = UIView()
    parentView.addSubview(childView)
    childView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
        childView.centerXAnchor.constraint(equalTo: parentView.centerXAnchor),
        childView.centerYAnchor.constraint(equalTo: parentView.centerYAnchor)
    ])
    ```

    - 基线对齐：对齐文本或其他元素的基线。

3. 关系约束（Relational Constraints）

    - 相等：使两个视图的特定属性（如宽度、高度或顶部对齐线）相等。

    ```swift
    let firstView = UIView()
    let secondView = UIView()
    NSLayoutConstraint.activate([
        secondView.widthAnchor.constraint(equalTo: firstView.widthAnchor)
    ])
    ```

    - 大于等于/小于等于：设置视图属性的最小或最大值，例如宽度大于等于或小于等于某个值。

    ```swift
    let view = UIView()
    view.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
        view.widthAnchor.constraint(greaterThanOrEqualToConstant: 100),
        view.heightAnchor.constraint(lessThanOrEqualToConstant: 200)
    ])
    ```

4. 安全区域和布局边界

    - 安全区域约束：确保内容不被顶部的刘海、底部的操作条或其他界面元素遮挡。

    ```swift
    let parentView = UIView()
    let childView = UIView()
    parentView.addSubview(childView)
    childView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
        childView.topAnchor.constraint(equalTo: parentView.safeAreaLayoutGuide.topAnchor)
    ])
    ```

    - 参照布局边界：将视图的边缘与父视图的布局边界对齐。
