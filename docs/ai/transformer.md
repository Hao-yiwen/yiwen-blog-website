# Transformer基础架构

Transformer是一种基于自注意力机制的神经网络架构，作为现代大语言模型的基础架构。它通过多头注意力机制和位置编码来捕获序列中的长距离依赖关系，能够有效处理自然语言处理中的各类任务。

## 基础架构

- 输入处理
    - Token分词：将输入文本切分成子词单元
    - Embedding层：将token转换为密集向量表示
    - 位置编码：注入位置信息以保持序列顺序

- 编码器(Encoder)
    - 多头自注意力层(Multi-head Self-attention)
        - 通过Query、Key、Value三个矩阵计算注意力分数
        - 使用scaled dot-product attention计算注意力权重
        - 应用softmax进行归一化
        - 使用注意力掩码防止信息泄露
    - 残差连接(Residual Connection)和层归一化(Layer Normalization) 
    - 前馈神经网络(Feed Forward Network)
        - 两层全连接层
        - 第一层使用ReLU激活函数
        - 残差连接和层归一化

- 解码器(Decoder) 
    - 第一个多头自注意力层
        - 处理已生成的输出序列
        - 使用掩码确保自回归生成
    - 第二个多头交叉注意力层
        - 将编码器的输出作为Key和Value
        - 解码器的隐状态作为Query
    - 每层后都有残差连接和层归一化
    - 最后是前馈神经网络层

## 工作原理

1. 输入处理：文本首先被分词并转换为词嵌入向量，同时添加位置编码信息

2. 编码过程：
   - 输入序列通过多层编码器块进行处理
   - 每一层都会通过自注意力机制捕获序列内的依赖关系
   - 通过残差连接和前馈网络进一步提取特征

3. 解码过程：
   - 解码器通过掩码自注意力层处理当前生成的序列
   - 通过交叉注意力层融合编码器的信息
   - 最终通过线性层和softmax生成下一个token的概率分布

通过这种架构设计，Transformer能够并行处理输入序列，高效捕获长距离依赖关系，并在各种NLP任务上取得优异性能。大规模预训练后，模型能够学习到丰富的语言知识和模式，从而能够理解和生成高质量的文本。

## 代码示例

https://github.com/Hao-yiwen/deeplearning/blob/master/pytorch/week3/practise_10_transformer.ipynb

## 架构升级

1. Flash Attention
   - 2022年提出的高效注意力计算机制
   - 通过优化内存访问模式减少IO开销
   - 显著降低内存占用和计算延迟
   - 在长序列任务中表现优异

2. GPT系列架构创新
   - 仅使用解码器的自回归模型
   - 通过因果掩码实现单向注意力
   - 支持大规模预训练和零样本学习
   - 在生成任务中表现出色

3. 其他优化技术
   - Sparse Attention：稀疏注意力机制
   - Linear Attention：线性复杂度的注意力计算
   - Rotary Position Embedding：旋转位置编码
   - Mixture of Experts：专家混合系统
