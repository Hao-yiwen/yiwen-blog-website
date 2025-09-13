# RAG（检索增强生成）原理文档

## 一、核心概念

**RAG = Embedding（语义编码） + 向量检索 + LLM生成**

将知识库文档转换为语义向量，通过余弦相似度找到最相关内容，提供给LLM生成准确答案。

## 二、工作原理

### 2.1 数学基础
```
余弦相似度 = A·B / (|A|×|B|)
```
- 衡量两个向量的方向相似性
- 结果范围：[-1, 1]，越接近1越相似
- 不受向量长度影响，只看语义方向

### 2.2 语义理解
```python
# 传统匹配：依赖关键词
"这车真快" vs "这辆汽车速度很高"  → 相似度低（词汇不同）

# 语义匹配：理解含义  
embed("这车真快") ≈ embed("这辆汽车速度很高")  → 相似度高（语义相同）
```

## 三、RAG流程

### 3.1 知识库构建（离线）
```python
文档切分 → Embedding编码 → 存入向量数据库

documents = ["诺兰是导演...", "盗梦空间讲述..."]
embeddings = embed_model(documents)  # 转为向量（如1536维）
vector_db.store(embeddings, documents)
```

### 3.2 检索生成（在线）
```python
用户查询 → 编码 → 检索相似文档 → 生成答案

query_embedding = embed_model(query)
similar_docs = vector_db.search(query_embedding, top_k=3)  # 余弦相似度
answer = LLM.generate(query, context=similar_docs)
```

## 四、为什么有效？

### 4.1 预训练模型的语义能力
- 在海量数据上学习，理解词汇、句子、上下文的语义关系
- 相似语义 → 相似向量 → 高余弦相似度

### 4.2 RAG的优势
- **实时性**：无需重训练，直接更新知识库
- **准确性**：基于真实文档，减少LLM幻觉
- **可解释**：能追溯信息来源

## 五、简单实现

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRAG:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, docs):
        """构建知识库"""
        self.documents = docs
        self.embeddings = self.embed_model.encode(docs)
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        # 1. 编码查询
        query_emb = self.embed_model.encode([query])
        
        # 2. 计算余弦相似度
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        
        # 3. 返回最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]
```

## 六、与推荐系统的类比

| 系统 | 编码对象 | 目标 | 核心算法 |
|-----|---------|------|---------|
| 电影推荐 | 电影特征 | 找相似电影 | 余弦相似度 |
| RAG | 文本段落 | 找相关知识 | 余弦相似度 |

**本质相同**：都是在高维空间中寻找向量夹角最小（最相似）的内容。

## 七、关键要点

1. **Embedding是核心**：将文本转换为包含语义信息的向量
2. **余弦相似度是桥梁**：连接查询和知识库
3. **语义理解是基础**：预训练模型提供的语义理解能力使一切成为可能

---

**总结**：RAG通过语义向量化和相似度计算，让LLM能够准确检索和利用外部知识，是解决LLM知识更新和幻觉问题的有效方案。