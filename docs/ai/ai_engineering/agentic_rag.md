---
title: Agentic RAG 高级检索方案
sidebar_position: 1
tags: [RAG, AI, Agent, Elasticsearch]
---

import rag_1 from "@site/static/img/rag_1.png";
import rag_2 from "@site/static/img/rag_2.png";
import rag_3 from "@site/static/img/rag_3.png";
import rag_4 from "@site/static/img/rag_4.png";
import rag_5 from "@site/static/img/rag_5.png";

# Agentic RAG 高级检索方案

## 核心理念

采用"三工具协同"的 Agentic 设计，AI 自主决策调用哪个工具。

<img src={rag_1} alt="AI Agent 工具箱" style={{width: '100%', display: 'block', margin: '20px auto'}} />

## 三工具职责

| 工具 | 功能 | 返回内容 | 使用场景 |
|------|------|---------|---------|
| `search_documents` | 文档级搜索 | 文档摘要列表 + doc_id | 了解有哪些相关文档 |
| `search_chunks` | Chunk 级混合搜索 | Chunk 列表 + doc_id | 搜索具体内容片段 |
| `get_document` | 获取完整文档 | 完整文档内容 | 需要完整上下文时 |

## 系统架构

<img src={rag_2} alt="系统架构" style={{width: '100%', display: 'block', margin: '20px auto'}} />

### search_documents 架构（文档级检索）

<img src={rag_3} alt="search_documents 内部架构" style={{width: '100%', display: 'block', margin: '20px auto'}} />

三路检索说明：

- **关键词精确匹配:** 用户输入的技术术语直接命中 LLM 提取的关键词
- **摘要 BM25 检索:** IK 分词后的词频匹配，适合中文关键词查询
- **摘要向量搜索:** 真正的语义搜索，理解查询意图匹配相似摘要

### search_chunks 架构（混合检索）

<img src={rag_4} alt="search_chunks 内部架构" style={{width: '100%', display: 'block', margin: '20px auto'}} />

## AI 调用流程示例

<img src={rag_5} alt="AI 调用流程示例" style={{width: '100%', display: 'block', margin: '20px auto'}} />

## 与现有系统对比

| 维度 | 现有向量检索 | Agentic 三工具 |
|------|-----------|--------------|
| 检索粒度 | 仅 Chunk | 文档 + Chunk |
| AI 自主性 | 被动接收结果 | 主动选择工具 |
| 上下文完整性 | Chunk 碎片化 | 可获取完整文档 |
| 工具数量 | 1 个 | 3 个各司其职 |

## 核心优势

1. **Agentic:** AI 自主决策工具调用，而非被动接收
2. **粒度可控:** 从文档级到 Chunk 级，按需获取
3. **上下文完整:** 通过 `get_document` 消除碎片化
4. **双索引:** 文档索引 + Chunk 索引，各自优化

最终实现：**发现 → 定位 → 补充 → 完整回答**
