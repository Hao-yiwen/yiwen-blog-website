---
title: Langfuse 核心概念与实战
sidebar_position: 2
tags: [Langfuse, LLM, 可观测性, RAG]
---

# Langfuse 核心概念与实战详细文档

## 1. 核心数据架构（从宏观到微观）

理解 Langfuse 的第一步，是理解它如何对 AI 应用的数据进行分层。我们可以用大家熟悉的 ChatGPT 网页版来类比：

- **👤 User（用户）**：**最高维度**。代表一个真实的物理用户（例如：张三，`user_id="user_zhangsan_001"`）。Langfuse 会自动为用户构建画像，统计 Token 消耗、会话数量、使用成本和平均反馈评分。
- **💬 Session（会话）**：代表一次连续的对话（例如：ChatGPT 左侧的某个对话线程）。一个 User 可以有多个 Session。
- **🔗 Trace（追踪）**：代表一次完整的请求-响应周期（例如：用户发送一条消息到收到回复）。一个 Session 可以包含多个 Trace。
- **🔍 Observation（观测）**：Trace 内部的具体步骤，分为三种类型：
  - **Span**：普通逻辑跨度（如：检索文档、数据预处理）
  - **Generation**：大模型调用（专门记录模型名称、Token 消耗、耗时等）
  - **Event**：离散事件（如：日志记录）

层级关系：`User → Session → Trace → Observation (Span / Generation / Event)`

## 2. 环境配置

```bash
pip install langfuse
```

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

## 3. 全链路实战代码：如何在一个请求中串联所有维度

使用 `@observe` 装饰器搭配 `langfuse_context`，可以在不破坏原有业务逻辑的情况下，优雅地注入所有维度信息。

```python
import uuid
import time
from langfuse.decorators import observe, langfuse_context

# 模拟前端传来的上下文信息
CURRENT_USER_ID = "user_zhangsan_001"
CURRENT_SESSION_ID = "session_chat_thread_999"

@observe()
def handle_user_request(user_message: str):
    """
    这是系统接收请求的入口。
    Langfuse 会自动为这个函数生成一个 Trace。
    我们要在这里把 User, Session 等高维度信息绑定到这个 Trace 上。
    """
    # 1. 绑定高维度上下文（User, Session, 标签等）
    langfuse_context.update_current_trace(
        name="rag_chat_turn",
        user_id=CURRENT_USER_ID,
        session_id=CURRENT_SESSION_ID,
        tags=["production", "rag-v2"],
        metadata={"source": "web_app"}
    )

    # 2. 检索文档（Span）
    docs = retrieve_documents(user_message)

    # 3. 调用大模型（Generation）
    reply = generate_reply(user_message, docs)

    # 4. 获取当前 Trace 的 ID，方便后续用于评分
    trace_id = langfuse_context.get_current_trace_id()

    return reply, trace_id

@observe(as_type="span")  # 标记为普通逻辑跨度
def retrieve_documents(query: str):
    """模拟向量数据库检索（RAG）"""
    time.sleep(0.5)
    return ["知识库文档1：Langfuse 很好用", "知识库文档2：Python 是好语言"]

@observe(as_type="generation")  # 标记为大模型生成，专门记录大模型指标
def generate_reply(query: str, context_docs: list):
    """模拟调用 LLM"""
    time.sleep(1)

    # 手动记录 Token 消耗（如果不使用 Langfuse 原生的 OpenAI 封装）
    langfuse_context.update_current_observation(
        model="gpt-4o",
        usage={
            "input": 150,
            "output": 50,
            "unit": "TOKENS"
        }
    )
    return "综合知识库内容，为您生成的回答..."

# 执行
reply, trace_id = handle_user_request("Langfuse 怎么用？")
print(f"回复: {reply}")
print(f"Trace ID: {trace_id}")
```

## 4. 用户反馈评分（Score）

当用户点击"👍 赞"或"👎 踩"时，前端将 `trace_id` 和反馈传回后端，可以这样记录：

```python
from langfuse import Langfuse

# 初始化客户端用于发送评分
langfuse_client = Langfuse()

def log_user_feedback(trace_id: str, is_like: bool):
    """记录用户反馈分数"""
    score_value = 1 if is_like else 0

    langfuse_client.score(
        trace_id=trace_id,       # 绑定到具体的 Trace
        name="user_thumbs_up",   # 评分维度的名称
        value=score_value,       # 具体分值
        comment="回答速度很快！"   # 可选附加评论
    )
    print(f"已记录 Trace {trace_id} 的用户评分: {score_value}")

# 模拟记录刚才那次请求的评分
log_user_feedback(trace_id, is_like=True)
```

## 5. 进阶：使用云端提示词（Prompt Management）

将 Prompt 与代码解耦，在 Langfuse UI 修改即可生效。

```python
from langfuse import Langfuse

langfuse_client = Langfuse()

def get_system_prompt():
    # 从云端拉取名为 "chat_system_prompt" 的提示词
    prompt = langfuse_client.get_prompt("chat_system_prompt")

    # 编译提示词（替换变量）
    final_prompt_string = prompt.compile(company="MyCorp")

    return final_prompt_string

# 在 LLM 调用中，将拉取到的 prompt 作为 system message 传入
```

## 总结

这套架构可以满足绝大部分 LLM 应用的工程化需求。掌握了 `User → Session → Trace → Observation` 这个数据结构，就能在 Langfuse 的面板里随心所欲地切块分析数据。
