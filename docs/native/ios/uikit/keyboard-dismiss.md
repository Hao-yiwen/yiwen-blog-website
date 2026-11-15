---
title: 键盘消失方法
sidebar_label: 键盘消失方法
date: 2024-06-25
last_update:
  date: 2024-06-25
---

# 键盘消失方法

在有输入框的时候，需要点击屏幕其他地方消失键盘和添加完成键。

现在将其作为基类，这样其他有textfield的页面直接添加基类就可以使用了。


```swift
import UIKit

class KeyboardDismissBaseController: UIViewController, UITextFieldDelegate {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        
        let tap = UITapGestureRecognizer(target: self, action: #selector(keyboardDismiss))
        view.addGestureRecognizer(tap)
    }
    
    // 添加完成键
    func addCompleteButton(textField: UITextField) {
        let toolbar = UIToolbar()
        toolbar.sizeToFit()
        
        let flexSpace = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        let doneButton = UIBarButtonItem(title: "完成", style: .done, target: self, action: #selector(doneButtonAction))
        
        toolbar.setItems([flexSpace, doneButton], animated: false)
        textField.inputAccessoryView = toolbar
    }
    
    // 完成键的动作
    @objc func doneButtonAction() {
        view.endEditing(true)
    }
    
    @objc func keyboardDismiss() {
        view.endEditing(true)
    }
    
    // UITextFieldDelegate 方法，按下 return 键时关闭键盘
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }
}
```