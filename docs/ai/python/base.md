# python标准库

Python 提供了非常丰富的标准库，覆盖了从基础数据操作到网络编程、文件处理等众多功能。这些标准库不需要额外安装，可以直接在 Python 中使用。以下是一些常用的标准库分类和具体模块

1. 数据类型与数据结构
	-	collections：提供了高级数据结构，如 deque、Counter、defaultdict。
	-	array：处理基本类型数组。
	-	heapq：实现堆（优先队列）操作。
	-	bisect：二分法查找和插入。
	-	queue：实现多种队列（如 FIFO、LIFO）。
	-	dataclasses：提供数据类支持，简化类的定义。

2. 文件与目录操作
	-	os：提供操作系统接口（文件、目录、环境变量）。
	-	shutil：高级文件操作，如复制、删除。
	-	pathlib：对象化的路径操作（推荐代替 os.path）。
	-	glob：查找符合特定模式的文件。
	-	fnmatch：文件名模式匹配。
	-	fileinput：方便地读取多个文件。
	-	tempfile：创建临时文件和目录。

3. 时间与日期
	-	time：处理时间戳、睡眠操作。
	-	datetime：更强大的日期和时间处理功能。
	-	calendar：日历操作。
	-	zoneinfo：时区处理（Python 3.9+）。

4. 数学与统计
	-	math：提供数学函数（如三角函数、对数、指数）。
	-	random：生成随机数和随机操作。
	-	statistics：计算均值、中位数、方差等统计数据。
	-	fractions：支持分数运算。
	-	decimal：支持高精度浮点数运算。

5. 字符串操作
	-	string：包含字符串操作的工具集合。
	-	re：正则表达式支持。
	-	textwrap：格式化文本。
	-	difflib：比较文本差异。
	-	unicodedata：处理 Unicode 数据。

6. 系统与进程
	-	sys：提供与 Python 解释器交互的功能。
	-	subprocess：运行外部命令并与之交互。
	-	argparse：命令行参数解析。
	-	logging：日志记录。
	-	getopt：简化版的命令行参数解析。
	-	multiprocessing：并行多进程支持。
	-	threading：多线程支持。

7. 网络编程
	-	socket：底层网络通信支持。
	-	http：HTTP 协议支持，包含服务器和客户端实现。
	-	urllib：处理 URL 的工具集合。
	-	json：JSON 数据的解析与生成。
	-	xml.etree.ElementTree：处理 XML 数据。
	-	email：处理电子邮件内容。
	-	asyncio：支持异步 IO 操作。

8. 数据存储与序列化
	-	pickle：序列化 Python 对象。
	-	sqlite3：轻量级内置数据库支持。
	-	csv：处理 CSV 文件。
	-	configparser：处理配置文件。
	-	plistlib：处理 macOS 的 plist 文件。

9. 测试与调试
	-	unittest：单元测试框架。
	-	doctest：在文档中直接写测试。
	-	traceback：打印异常信息。
	-	pdb：Python 的调试器。
	-	timeit：测量代码运行时间。

10. 安全与加密
	-	hashlib：提供哈希算法（如 SHA256、MD5）。
	-	hmac：实现 HMAC 算法。
	-	ssl：处理 SSL/TLS 加密通信。
	-	secrets：生成密码、安全令牌等。

11. 其他实用工具
	-	itertools：迭代器工具集。
	-	functools：函数工具（如 lru_cache、partial）。
	-	operator：高效执行操作符。
	-	typing：支持类型注解。
	-	enum：定义枚举类型。
	-	base64：Base64 编码和解码。
	-	uuid：生成唯一标识符。

12. 图形界面
	-	tkinter：Python 的内置 GUI 工具库。

13. 开发和调试
	-	inspect：获取对象的内部信息。
	-	ast：处理 Python 抽象语法树。
	-	code：动态执行 Python 代码。
	-	venv：创建虚拟环境。

```py

# 1. 数据类型与数据结构

from collections import Counter

# 统计字符出现次数
text = "hello world"
counter = Counter(text)
print(counter)  # 输出：Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# 2. 文件与目录操作

import os
from pathlib import Path

# 获取当前目录
print(os.getcwd())

# 使用 pathlib 创建文件
file = Path("example.txt")
file.write_text("Hello, pathlib!")
print(file.read_text())  # 输出：Hello, pathlib!

# 3. 时间与日期

from datetime import datetime

# 当前日期和时间
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))  # 输出：2025-01-20 12:34:56

# 4. 数学与统计

import math
import random

# 计算平方根
print(math.sqrt(16))  # 输出：4.0

# 生成随机数
print(random.randint(1, 10))  # 输出：随机整数 1 到 10

# 5. 字符串操作

import re

# 正则表达式匹配
pattern = r"\b\w{5}\b"
text = "hello world python"
matches = re.findall(pattern, text)
print(matches)  # 输出：['hello', 'world']

# 6. 系统与进程

import subprocess

# 执行外部命令
result = subprocess.run(["echo", "Hello from subprocess"], capture_output=True, text=True)
print(result.stdout)  # 输出：Hello from subprocess

# 7. 网络编程

import socket

# 创建一个简单的 TCP/IP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket created successfully!")
sock.close()

# 8. 数据存储与序列化

import pickle

# 序列化和反序列化
data = {"name": "Alice", "age": 25}
serialized = pickle.dumps(data)
print(pickle.loads(serialized))  # 输出：{'name': 'Alice', 'age': 25}

# 9. 测试与调试

import unittest

# 定义测试用例
class TestMathOperations(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)

# 运行测试
if __name__ == "__main__":
    unittest.main()

# 10. 安全与加密

import hashlib

# 计算 SHA256 哈希
data = "hello"
hash_object = hashlib.sha256(data.encode())
print(hash_object.hexdigest())  # 输出：哈希值

# 11. 其他实用工具

from itertools import permutations

# 生成排列
items = [1, 2, 3]
for perm in permutations(items):
    print(perm)

# 12. 图形界面

import tkinter as tk

# 创建简单的 GUI 窗口
root = tk.Tk()
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()
root.mainloop()

# 13. 开发和调试

import inspect

# 获取函数的签名
def example_func(a, b):
    return a + b

print(inspect.signature(example_func))  # 输出：(a, b)
```