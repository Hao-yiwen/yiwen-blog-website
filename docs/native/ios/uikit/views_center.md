---
title: 如何将多个组件一起设置居中
sidebar_label: 如何将多个组件一起设置居中
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 如何将多个组件一起设置居中

今天有一个场景，需要将label和输入框一起居中在屏幕中间，但是找了一圈发现并不能直接设置两个组件一起水平居中，这样并不会生效。

~~于是在查询很多信息后，发现两种解决方案。~~

经过实际时间，发现还是纯代码+stackview方式比较快，interface build会极大的削减开发速度。

```swift
class CenterController: UIViewController {
    @IBOutlet weak var stackview: UIStackView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 创建 label
        let label = UILabel()
        label.text = "name"
        
        // 创建 text field
        let textInput = UITextField()
        textInput.placeholder = "请输入 name"
        textInput.borderStyle = .roundedRect
        
        // 创建一个新的 stack view
        let innerStackView = UIStackView(arrangedSubviews: [label, textInput])
        innerStackView.axis = .vertical
        innerStackView.alignment = .center
        innerStackView.spacing = 20
        innerStackView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(innerStackView)
        
        // 设置约束
        NSLayoutConstraint.activate([
            // 将新的 stack view 水平居中
            innerStackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            
            // 将新的 stack view 的顶部距离原来的 stack view 20 点
            innerStackView.topAnchor.constraint(equalTo: stackview.bottomAnchor, constant: 20)
        ])
    }
}
```