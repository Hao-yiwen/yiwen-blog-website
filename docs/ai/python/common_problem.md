# 常见问题

## __pycache__ 是什么？

**Python 的字节码缓存目录**，用来加速程序运行。（git提交建议删除）

### 为什么会有它？

```python
# 第一次运行 main.py
python main.py

# Python 做了什么：
# 1. 读取 main.py（源码 .py）
# 2. 编译成字节码（.pyc）
# 3. 保存到 __pycache__/main.cpython-311.pyc
# 4. 执行字节码

# 第二次运行
python main.py
# 1. 检查 main.py 有没有改动
# 2. 没改？直接用缓存的 .pyc（快！）
# 3. 改了？重新编译
```
