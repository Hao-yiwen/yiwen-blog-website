---
title: 文档分块 (Chunking) 逻辑设计
sidebar_position: 3
tags: [RAG, chunking, NLP]
---

# 文档分块 (Chunking) 逻辑设计说明

在 RN Doc Search 系统中，我们针对 React Native 中文文档（Markdown 格式）实现了一套定制化的分块（Chunking）逻辑。目标是**保证代码块的完整性**，**避免语义断裂**，以及**减少无效的碎片块**。

分块逻辑主要实现在 `pipeline/chunk_processor.py` 的 `ChunkProcessor` 类中。

## 1. 核心设计原则

1. **结构感知**：优先按 Markdown 的标题层级和段落结构切分，而不是生硬地按 Token 数截断。
2. **代码块不截断**：无论代码块有多长，都尽量保持完整，这对于基于代码内容的搜索和理解至关重要。
3. **消除碎片**：孤立的标题或过短的句子会被合并到相邻的正文中，避免出现只有几个字或单个标题的无意义 Chunk。

## 2. 分块流水线 (Pipeline)

整个文档分块过程分为以下几个阶段：

### 阶段一：Markdown 节点解析 (MarkdownNodeParser)
使用 LlamaIndex 的 `MarkdownNodeParser` 获取初始节点。这一步会根据 Markdown 的标题层级（Heading 级别）进行初步的结构化切分，保留文档的骨架。

### 阶段二：细粒度语义块拆分 (Split into Blocks)
对阶段一获取到的每个大节点的文本，进一步按**行**、**段落**和**代码块**进行拆分，生成一系列基础的语义块（Blocks）。
- 遇到 ` ``` ` 开始的 fenced code block 时，将其完整提取为一个独立的 Block。
- 遇到 Markdown 标题或空行时，也会将累积的文本刷新为一个独立的 Block。

### 阶段三：短块与标题合并 (Merge Short Blocks)
遍历所有的 Blocks，进行合并优化：
- **纯标题块**（只有标题没有内容的块）必须与后文合并。
- **过短块**（长度小于 `min_chunk_chars`，目前为 120 字符），如果不是完整的代码块，则尝试并入相邻的文本块。
- 这一步有效避免了最终落盘时产生大量碎片化的独立 Chunk。

### 阶段四：处理超长块 (Split Large Blocks)
对于超过软性长度上限（`max_chunk_chars`，默认 2048）的块进行拆分：
- **完整代码块**：**特例豁免**，按原样保留，坚决不进行拆分截断。
- **普通文本块**：优先按换行符（行）进行拆分。
- **超长单行**：如果单行文本仍超限，则按中文/英文的标点符号（`。！？；.!?;`）进行**句子级拆分**。
- **兜底策略**：如果连单句都超过最大长度，才进行字数硬切分（Hard Wrap）。

### 阶段五：打包生成最终 Chunks (Pack into Chunks)
将经过上述处理的 Blocks 按顺序拼接打包：
- 将多个 Block 用空行连接，不断累加，直到达到长度上限 `max_chunk_chars`。
- 如果当前累加文本的长度不足（且不是纯代码块），会强制继续向后合并下一个 Block。
- 最终再次进行尾部碎片的合并清理，形成最终的 Chunk 文本列表。

### 阶段六：代码特征提取 (Extract Code Blocks)
对于每个生成的最终 Chunk 文本，使用正则表达式提取出其中所有的代码片段，将其拼接后单独存储在 `code_blocks` 字段中。该字段在后续会被送入针对代码特征优化的 BM25 索引（使用 CamelCase 和下划线分词），以强化代码搜索。

### 阶段七：向量化 (Embedding)
将每个 Chunk 的正文内容送入 OpenAI Embedding 模型（分批调用），生成 3072 维的 `content_vector` 向量表示。

## 3. 核心参数配置

- `max_chunk_chars`: 单个 Chunk 的最大字符数软上限（非强制，代码块可突破此限制）。默认为 `max(CHUNK_SIZE * 4, 2048)`。
- `min_chunk_chars`: Chunk 的最小字符数阈值（设定为 120 字符）。小于此长度且非代码块的内容会被强制与相邻块合并。

## 4. 输出数据结构 (ChunkRecord)

分块完成后，每篇文档会被转换为多个 `ChunkRecord`，准备写入存储层：

```python
@dataclass
class ChunkRecord:
    chunk_id: str               # Chunk 唯一标识，如 "doc_id_chunk_0"
    doc_id: str                 # 关联的文档 ID
    content: str                # Chunk 最终的正文内容
    code_blocks: str            # 提取出的纯代码文本（用于专用的代码 BM25 检索）
    content_vector: List[float] # Chunk content 的向量 (OpenAI Embedding)
    title: str                  # 文档的原始标题
```

## 5. 总结
这种**结构化切分 + 短块合并 + 代码感知 + 标点断句兜底**的多级处理策略，相比简单的滑动窗口或纯 Token 切分（如 `TokenTextSplitter`），取得了更好的检索上下文质量。它确保了向量检索时，相关代码段不会被从中腰斩，同时能为纯代码特征抽取提供最高质量的数据源。
