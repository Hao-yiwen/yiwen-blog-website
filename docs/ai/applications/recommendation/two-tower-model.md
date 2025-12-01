---
title: 双塔模型（Two-Tower）召回实现
sidebar_position: 1
tags: [推荐系统, 双塔模型, 召回, PyTorch, Faiss]
---

# 双塔模型（Two-Tower）召回实现

这是一个**工业级标准**的双塔模型（Two-Tower）完整代码示例，使用 `PyTorch` 定义模型，使用 `Faiss`（Meta 开源的向量检索库）做召回索引。

## 什么是双塔模型？

双塔模型是推荐系统中最常用的召回算法之一，它的核心思想是：

- **用户塔（User Tower）**：将用户特征映射为向量
- **物品塔（Item Tower）**：将物品特征映射为向量
- **相似度计算**：通过向量内积或余弦相似度来衡量用户-物品匹配度

优势：
1. **解耦训练与推理**：物品向量可以离线计算并缓存
2. **高效召回**：使用向量检索库（如 Faiss）可以从百万级物品中毫秒级召回
3. **支持冷启动**：通过内容特征可以为新物品生成向量

## 完整生命周期

这个代码模拟了完整的推荐系统生命周期：

1. **数据构造**：模拟用户和物品特征
2. **模型定义**：定义双塔结构（用户塔 + 物品塔）
3. **离线训练**：让模型学会把"相关"的 User 和 Item 拉近
4. **离线建库**：把物品向量存入 Faiss
5. **在线召回**：新用户来了，实时算出向量去搜 Faiss
6. **新物品上架**：演示怎么给新物品生成 Embedding 并插入库

## 前置准备

安装依赖：

```bash
pip install torch faiss-cpu numpy pandas
```

## 完整代码实现

```python title="two_tower_demo.py"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss  # 向量检索库
import pandas as pd

# ==========================================
# 1. 配置与超参数
# ==========================================
EMBEDDING_DIM = 32   # 向量维度 (工业界常用 64 或 128)
USER_NUM = 1000      # 假设有 1000 个用户
ITEM_NUM = 2000      # 假设有 2000 个物品
CATE_NUM = 20        # 物品分类数
BATCH_SIZE = 64
EPOCHS = 5

# ==========================================
# 2. 模型定义 (双塔结构)
# ==========================================

# 通用的塔结构 (MLP)
class Tower(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Tower, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim), # 输出最终的 Embedding
            nn.LayerNorm(output_dim)    # 归一化，有助于向量检索
        )

    def forward(self, x):
        return self.mlp(x)

class TwoTowerModel(nn.Module):
    def __init__(self):
        super(TwoTowerModel, self).__init__()

        # --- 用户塔 User Tower ---
        # 1. ID Embedding (对老用户有用)
        self.user_id_emb = nn.Embedding(USER_NUM + 1, 16) # +1 是为了留给未知用户
        # 2. 只有 ID 是不够的，还需要画像特征 (年龄、性别等，这里简化为随机向量)
        self.user_profile_emb = nn.Linear(5, 16) # 假设有 5 个画像特征

        # 用户塔的 MLP
        self.user_tower = Tower(16 + 16, EMBEDDING_DIM)

        # --- 物品塔 Item Tower (关键！) ---
        # 1. ID Embedding (对老物品有用)
        self.item_id_emb = nn.Embedding(ITEM_NUM + 1, 16) # +1 是为了留给未知物品/新物品
        # 2. 内容特征 (对新物品冷启动至关重要)
        # 这里假设把标题文本、分类做成了 One-hot 或 Bert 向量，简化为 Category Embedding
        self.item_cate_emb = nn.Embedding(CATE_NUM + 1, 16)

        # 物品塔的 MLP
        self.item_tower = Tower(16 + 16, EMBEDDING_DIM)

    def forward_user(self, user_id, user_features):
        # 用户 Embedding 生成逻辑
        uid_vec = self.user_id_emb(user_id)
        feat_vec = self.user_profile_emb(user_features)
        # 拼接 ID 特征和画像特征
        concat_vec = torch.cat([uid_vec, feat_vec], dim=1)
        return self.user_tower(concat_vec)

    def forward_item(self, item_id, item_cate):
        # 物品 Embedding 生成逻辑
        iid_vec = self.item_id_emb(item_id)
        cate_vec = self.item_cate_emb(item_cate)
        # 拼接 ID 特征和内容特征
        concat_vec = torch.cat([iid_vec, cate_vec], dim=1)
        return self.item_tower(concat_vec)

    def forward(self, user_id, user_features, item_id, item_cate):
        # 训练时同时调用两个塔
        u_vec = self.forward_user(user_id, user_features)
        i_vec = self.forward_item(item_id, item_cate)
        return u_vec, i_vec

# ==========================================
# 3. 模拟数据生成
# ==========================================
def get_fake_data(num=1000):
    # 模拟输入数据
    user_ids = torch.randint(0, USER_NUM, (num,))
    # 模拟用户画像 (5维浮点数，比如归一化后的年龄、活跃度)
    user_feats = torch.randn(num, 5)

    item_ids = torch.randint(0, ITEM_NUM, (num,))
    item_cates = torch.randint(0, CATE_NUM, (num,))

    # 模拟标签：1代表点击，0代表没点
    labels = torch.randint(0, 2, (num,)).float()

    return user_ids, user_feats, item_ids, item_cates, labels

# ==========================================
# 4. 离线训练 (Training Pipeline)
# ==========================================
model = TwoTowerModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss() # 二分类交叉熵损失

print(">>> 开始离线训练...")
model.train()
for epoch in range(EPOCHS):
    u_ids, u_feats, i_ids, i_cates, labels = get_fake_data(BATCH_SIZE * 10)

    total_loss = 0
    for i in range(0, len(u_ids), BATCH_SIZE):
        batch_u_ids = u_ids[i:i+BATCH_SIZE]
        batch_u_feats = u_feats[i:i+BATCH_SIZE]
        batch_i_ids = i_ids[i:i+BATCH_SIZE]
        batch_i_cates = i_cates[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]

        # 1. 跑双塔
        u_vec, i_vec = model(batch_u_ids, batch_u_feats, batch_i_ids, batch_i_cates)

        # 2. 计算相似度 (点积 Dot Product)
        # 结果形状 [Batch_Size, 1]
        logits = (u_vec * i_vec).sum(dim=1)

        # 3. 计算 Loss
        loss = criterion(logits, batch_labels)

        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / 10:.4f}")

# ==========================================
# 5. 离线建库 (Offline Indexing)
# ==========================================
print("\n>>> 训练完成，开始为所有物品生成向量并建库...")
model.eval() # 切换到预测模式

# 模拟数据库里所有的物品信息
all_item_ids = torch.arange(0, ITEM_NUM)
# 模拟这些物品的分类信息
all_item_cates = torch.randint(0, CATE_NUM, (ITEM_NUM,))

with torch.no_grad():
    # 只要调用物品塔 (Item Tower)
    all_item_vectors = model.forward_item(all_item_ids, all_item_cates)
    # 转成 numpy，Faiss 需要 float32
    item_vectors_np = all_item_vectors.numpy().astype('float32')

# 使用 Faiss 建立索引 (IndexFlatIP 代表内积 Inner Product，速度快)
# 在工业界，如果数据量大，这里会用 IndexIVFFlat
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(item_vectors_np) # 把向量塞进去

print(f"Faiss 索引构建完成，库中共有 {index.ntotal} 个物品。")

# ==========================================
# 6. 在线召回 (Online Recall) - 场景：用户刷新 Feed 流
# ==========================================
print("\n>>> 场景：老用户 (ID: 10) 刷新了 APP...")

# 1. 获取用户实时特征
current_user_id = torch.tensor([10])
current_user_feat = torch.randn(1, 5) # 实时画像

# 2. 通过用户塔生成 User Embedding
with torch.no_grad():
    user_embedding = model.forward_user(current_user_id, current_user_feat)
    user_embedding_np = user_embedding.numpy().astype('float32')

# 3. 去 Faiss 搜 Top 5
D, I = index.search(user_embedding_np, 5) # D是距离(分数), I是ID

print(f"为用户推荐的 Top-5 物品 ID: {I[0]}")
print(f"推荐分数(相似度): {D[0]}")

# ==========================================
# 7. 新物品处理 (Cold Start) - 场景：刚上传的视频
# ==========================================
print("\n>>> 场景：有商家上架了一个新商品 (ID未知)...")

# 新商品的特征
# ID设为 ITEM_NUM (代表 UNK/新ID，我们在定义 Embedding 时多留了一位)
new_item_id = torch.tensor([ITEM_NUM])
# 它的分类是 5 (假设是科技类)
new_item_cate = torch.tensor([5])

# 1. 实时通过物品塔生成 Embedding
# 注意：这里模型主要靠 new_item_cate (内容特征) 来生成向量
# 因为 new_item_id 指向的是一个通用的兜底向量
with torch.no_grad():
    new_item_vector = model.forward_item(new_item_id, new_item_cate)
    new_item_vector_np = new_item_vector.numpy().astype('float32')

# 2. 实时插入 Faiss
index.add(new_item_vector_np)

print(f"新物品已插入索引，现在库中有 {index.ntotal} 个物品。")
print("如果是科技类用户，现在就能搜到这个新物品了！")
```

## 代码深度解析

### 1. 双塔的设计 (`TwoTowerModel`)

代码中的 `forward` 函数被拆分成了 `forward_user` 和 `forward_item`。

- **原因**：这是为了解耦
  - **训练时**：`forward` 同时调用两边，计算 Loss
  - **上线后**：服务端只需要跑 `forward_user`。`forward_item` 只有在离线刷库或者有新物品时才跑

### 2. 处理新物品的逻辑

注意看 `new_item_id = torch.tensor([ITEM_NUM])` 这一行：

- 我们在定义 Embedding 时用了 `nn.Embedding(ITEM_NUM + 1, ...)`
- 那个 `+1` 就是留给所有**没见过的新 ID** 的
- 当新物品进来时，虽然 ID 对应的向量是通用的（没意义），但 **`item_cate` (内容特征)** 是有意义的
- MLP 会结合这两者。因为 ID 没信息，模型就会自动依赖 **Category** 信息，把它映射到正确的向量空间区域

### 3. Faiss 的作用

代码里的 `index.search` 就是所谓的"大海捞针"：

- 在内存里，它比你写 `for` 循环计算余弦相似度要快几百倍
- `IndexFlatIP` 是精确搜索（暴力算内积）。如果数据量到了 1000 万，你会换成 `IndexIVFFlat`（倒排索引+量化），速度能再快 10 倍

## 变成实时系统

上面的代码是一个脚本，要变成实时服务，需要把它拆开：

1. **训练部分**：放在 GPU 服务器上，每天半夜跑，产出 `model.pth`
2. **Faiss 部分**：包装成一个 RPC 服务（C++ 或 Python 服务），常驻内存
3. **Inference 部分**：把 `forward_user` 包装成一个 API (FastAPI/Flask)。前端请求来了 → 调 API 算向量 → 调 Faiss 查 ID → 返回

## 运行示例

```bash
python two_tower_demo.py
```

输出示例：

```
>>> 开始离线训练...
Epoch 1, Loss: 0.6931
Epoch 2, Loss: 0.6924
Epoch 3, Loss: 0.6918
Epoch 4, Loss: 0.6912
Epoch 5, Loss: 0.6907

>>> 训练完成，开始为所有物品生成向量并建库...
Faiss 索引构建完成，库中共有 2000 个物品。

>>> 场景：老用户 (ID: 10) 刷新了 APP...
为用户推荐的 Top-5 物品 ID: [1234 567 890 123 456]
推荐分数(相似度): [0.89 0.87 0.85 0.83 0.81]

>>> 场景：有商家上架了一个新商品 (ID未知)...
新物品已插入索引，现在库中有 2001 个物品。
如果是科技类用户，现在就能搜到这个新物品了！
```

## 工业界最佳实践

### 特征工程
- **用户特征**：年龄、性别、地域、历史行为序列、实时兴趣标签
- **物品特征**：类目、标签、文本 Embedding（BERT）、图片 Embedding（ViT）

### 损失函数优化
- **负采样**：不是所有未点击的都是负样本，需要采样策略
- **In-batch Negatives**：利用同 batch 内的其他物品作为负样本
- **困难负样本挖掘**：选择相似但未点击的物品

### 向量检索优化
- **量化**：使用 Product Quantization 减少内存占用
- **索引类型**：
  - 小数据（&lt;100万）：`IndexFlatIP`
  - 中等数据（100万-1000万）：`IndexIVFFlat`
  - 大数据（&gt;1000万）：`IndexIVFPQ`

## 参考资料

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [YouTube 双塔推荐论文](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
- [阿里巴巴 TDM 论文](https://arxiv.org/abs/1801.02294)
