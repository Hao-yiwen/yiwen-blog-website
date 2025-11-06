---
title: FastAPI 入门指南
sidebar_label: FastAPI 入门指南
date: 2025-11-06
last_update:
  date: 2025-11-06
---

# FastAPI 入门指南

## 📖 什么是 FastAPI？

FastAPI 是一个现代、快速（高性能）的 Python Web 框架，用于构建 API。

**一句话概括：** 用 Python 类型提示快速构建高性能 API 的框架。

### 核心特点

- ⚡ **极快**：性能媲美 NodeJS 和 Go（基于 Starlette 和 Pydantic）
- 🚀 **开发效率高**：开发速度提升约 200%-300%
- 🐛 **更少 Bug**：减少约 40% 的人为错误
- 🧠 **智能**：极佳的编辑器支持，自动补全
- 📝 **自动文档**：自动生成交互式 API 文档
- 🔒 **类型安全**：基于 Python 类型提示
- ⚙️ **标准化**：基于 OpenAPI 和 JSON Schema（
  - **OpenAPI** 是一种描述 REST API 的标准格式，支持自动生成文档、代码和接口测试。
  - **JSON Schema** 是用来描述 JSON 数据结构的标准。FastAPI 利用 JSON Schema 自动生成数据校验规则和接口文档，保证数据格式的正确性。
  ）

---

## 🚀 快速开始

### 1. 安装

```bash
# 使用 pip
pip install "fastapi[standard]"

# 使用 uv（推荐）
uv add "fastapi[standard]"
```

### 2. 创建第一个 API

创建 `main.py` 文件：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

### 3. 运行服务器

```bash
# 开发模式（自动重载）
uvicorn main:app --reload

# 或使用 uv
uv run uvicorn main:app --reload
```

### 4. 访问

- **API 接口**: http://localhost:8000
- **交互式文档 (Swagger UI)**: http://localhost:8000/docs
- **替代文档 (ReDoc)**: http://localhost:8000/redoc

---

## 🎯 核心概念

### 1. 路径操作（Path Operations）

FastAPI 使用装饰器定义 API 端点：

```python
from fastapi import FastAPI

app = FastAPI()

# GET 请求
@app.get("/users")
def get_users():
    return {"users": ["Alice", "Bob"]}

# POST 请求
@app.post("/users")
def create_user(name: str):
    return {"message": f"User {name} created"}

# PUT 请求
@app.put("/users/{user_id}")
def update_user(user_id: int, name: str):
    return {"user_id": user_id, "name": name}

# DELETE 请求
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    return {"message": f"User {user_id} deleted"}
```

### 2. 路径参数（Path Parameters）

从 URL 中获取参数：

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    # FastAPI 自动验证 user_id 必须是整数
    return {"user_id": user_id}

@app.get("/files/{file_path:path}")
def read_file(file_path: str):
    # :path 可以匹配包含斜杠的路径
    return {"file_path": file_path}
```

### 3. 查询参数（Query Parameters）

从 URL 查询字符串获取参数：

```python
# /items?skip=0&limit=10
@app.get("/items")
def get_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# 可选参数
@app.get("/search")
def search(q: str | None = None):
    if q:
        return {"query": q}
    return {"message": "No query provided"}
```

### 4. 请求体（Request Body）

使用 Pydantic 模型定义请求体：

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int | None = None

@app.post("/users")
def create_user(user: User):
    # FastAPI 自动验证和解析 JSON
    return {"message": f"User {user.name} created", "user": user}
```

### 5. 响应模型（Response Model）

定义 API 返回的数据结构：

```python
class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    # 不返回密码等敏感信息

@app.post("/users", response_model=UserResponse)
def create_user(user: User):
    # 即使返回了额外字段，也只会返回 UserResponse 定义的字段
    return {
        "id": 1,
        "name": user.name,
        "email": user.email,
        "password": "secret"  # 这个不会被返回
    }
```

---

## 📚 常见使用场景

### 1. RESTful API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# 模拟数据库
fake_db = {}

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

# 创建
@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in fake_db:
        raise HTTPException(status_code=400, detail="Item already exists")
    fake_db[item_id] = item
    return item

# 读取
@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]

# 更新
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    fake_db[item_id] = item
    return item

# 删除
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del fake_db[item_id]
    return {"message": "Item deleted"}
```

### 2. 数据验证

```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)
    password: str = Field(..., min_length=8)

@app.post("/register")
def register(user: User):
    # FastAPI 自动验证：
    # - username 长度 3-50
    # - email 格式正确
    # - age 在 0-150 之间
    # - password 至少 8 个字符
    return {"message": "User registered successfully"}
```

### 3. 文件上传

```python
from fastapi import File, UploadFile

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }
```

### 4. 异步处理

```python
import asyncio

@app.get("/slow")
async def slow_endpoint():
    # 异步执行，不会阻塞其他请求
    await asyncio.sleep(5)
    return {"message": "Finished after 5 seconds"}

@app.get("/fast")
async def fast_endpoint():
    # 其他请求可以在 slow_endpoint 等待时执行
    return {"message": "Fast response"}
```

### 5. 依赖注入

```python
from fastapi import Depends

# 依赖函数
def get_current_user(token: str):
    # 验证 token，返回用户
    if token != "secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"username": "alice"}

@app.get("/me")
def read_current_user(current_user: dict = Depends(get_current_user)):
    # current_user 由 get_current_user 提供
    return current_user
```

### 6. 数据库集成（SQLAlchemy）

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends

# 数据库设置
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 数据库模型
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

# Pydantic 模型
class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    
    class Config:
        from_attributes = True  # 可以从 ORM 对象创建

# 依赖：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API 端点
@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = UserDB(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
```

---

## 🔍 FastAPI vs 其他框架

### FastAPI vs Flask

| 特性 | FastAPI | Flask |
|------|---------|-------|
| 性能 | ⚡ 极快 | 🐌 较慢 |
| 异步支持 | ✅ 原生支持 | ⚠️ 需要额外配置 |
| 类型提示 | ✅ 完整支持 | ❌ 不支持 |
| 自动文档 | ✅ 自动生成 | ❌ 需要手动 |
| 数据验证 | ✅ 自动验证 | ❌ 需要手动 |
| 学习曲线 | 📈 中等 | 📉 简单 |
| 适合场景 | API、微服务 | Web 应用、小项目 |

**代码对比：**

```python
# FastAPI - 自动验证和文档
@app.post("/users")
def create_user(user: User):  # User 是 Pydantic 模型
    return user

# Flask - 需要手动验证
@app.route("/users", methods=["POST"])
def create_user():
    data = request.get_json()
    # 需要手动验证每个字段
    if "name" not in data:
        return {"error": "name is required"}, 400
    # ... 更多验证
    return data
```

### FastAPI vs Django

| 特性 | FastAPI | Django |
|------|---------|--------|
| 类型 | 🔌 微框架 | 🏢 全功能框架 |
| 性能 | ⚡ 极快 | 🚶 中等 |
| 学习曲线 | 📈 中等 | 📈 陡峭 |
| Admin 面板 | ❌ 无 | ✅ 内置 |
| ORM（对象关系映射，简化数据库操作，更安全高效） | ⚠️ 需手动集成第三方库 (如 SQLAlchemy、Tortoise ORM) | ✅ 内置（Django ORM，自动管理数据库迁移与模型） |
| 适合场景 | API、微服务 | 完整 Web 应用 |

---

## 🎨 项目结构推荐

```
my-fastapi-project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── models/              # Pydantic 模型
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── schemas/             # 数据库模型
│   │   ├── __init__.py
│   │   └── database.py
│   ├── routers/             # 路由
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── items.py
│   ├── dependencies.py      # 依赖
│   └── config.py            # 配置
├── tests/                   # 测试
│   └── test_main.py
├── pyproject.toml           # 项目配置
└── README.md
```

**示例代码：**

```python
# app/main.py
from fastapi import FastAPI
from app.routers import users, items

app = FastAPI(title="My API", version="1.0.0")

app.include_router(users.router)
app.include_router(items.router)

@app.get("/")
def root():
    return {"message": "Welcome to My API"}


# app/routers/users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
def get_users():
    return {"users": []}

@router.post("/")
def create_user():
    return {"message": "User created"}
```

---

## 💡 最佳实践

### 1. 使用环境变量

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My API"
    database_url: str
    secret_key: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. 错误处理

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in database:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "Custom header"}
        )
    return database[item_id]
```

### 3. CORS 配置

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. 后台任务

```python
from fastapi import BackgroundTasks

def send_email(email: str):
    # 发送邮件的耗时操作
    print(f"Sending email to {email}")

@app.post("/send-notification")
def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email, email)
    return {"message": "Email will be sent in background"}
```

---

## 📦 常用扩展

```bash
# 数据库
uv add sqlalchemy psycopg2-binary

# 环境变量管理
uv add python-dotenv pydantic-settings

# 认证
uv add python-jose passlib[bcrypt]

# 测试
uv add pytest httpx

# 性能优化
uv add "uvicorn[standard]"
```

---

## 🎓 学习资源

- **官方文档**: https://fastapi.tiangolo.com/
- **教程**: https://fastapi.tiangolo.com/tutorial/
- **GitHub**: https://github.com/tiangolo/fastapi
- **社区**: https://github.com/tiangolo/fastapi/discussions

---

## 🚀 总结

**FastAPI 适合：**
- ✅ 构建现代 REST API
- ✅ 微服务架构
- ✅ 需要高性能的应用
- ✅ 需要自动文档的项目
- ✅ 使用类型提示的 Python 开发者

**快速开始三步：**
1. `uv add "fastapi[standard]"`
2. 写几行代码定义 API
3. `uv run uvicorn main:app --reload`

现在就开始你的 FastAPI 之旅吧！🎉