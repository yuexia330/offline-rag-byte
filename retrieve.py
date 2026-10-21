import os
import json
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer,util
import ollama

#1.读数据
with open("data/chunks.json","r",encoding="utf-8") as f:
    chunks=json.load(f)

#我们要把文本内容和ID分别存进两个列表里
corpus=[]#存纯文本
chunk_ids=[]#存对应的ID

for item in chunks:
    corpus.append(item["content"])
    chunk_ids.append(item["id"])

# 2. 中文分词（极其关键的一步）
# BM25 的核心是算词频。英文直接按空格切，但中文不行。
# 所以我们要把 "北京抖音信息服务有限公司" 切分成 ["北京", "抖音", "信息", "服务", "有限公司"]
#定义一个迷你的停用词表
stop_words={'的', '是', '什么', '?', '，', '。', '了', '在', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
#在构建 tokenized_corpus 的时候过滤
tokenized_corpus=[]
for text in corpus:
    # jieba.lcut 是精确模式分词，返回一个列表,# jieba.lcut 切分后，遍历每一个词，如果不在停用词表里，且去空格后长度大于0，就保留
    words=jieba.lcut(text)
    filtered_words=[w for w in words if w not in stop_words and w.strip()!=""]
    tokenized_corpus.append(filtered_words)

# 3. 初始化 BM25 模型
# 把切好词的语料库喂给它，它就建好索引了
bm25=BM25Okapi(tokenized_corpus)

# 4. 读我造的问题集，拿第一个问题来试水
with open("data/test_questions.json","r",encoding="utf-8") as f:
    test_questions=json.load(f)

# 取第一个问题
query=test_questions[0]["question"]
print(f"\n正在检索的问题：{query}")

# 问题也要分词！不然 BM25 不知道拿什么去搜
tokenized_query=[w for w in jieba.lcut(query) if w not in stop_words and w.strip()!=""]
print(f"过滤后的问题分词结果：{tokenized_query}")

# 5. 开始算分
# get_scores 会返回一个列表，里面是语料库里每一个 chunk 算出来的分数
scores=bm25.get_scores(tokenized_query)

# 6. 排序找 Top 5（这一步最容易卡住）
# 我们需要把分数和 chunk_ids 绑在一起
# 把 (分数, ID) 组合成一个元组，放进列表里
score_id_pairs=[]
for i in range(len(scores)):
    score=scores[i]
    cid=chunk_ids[i]
    score_id_pairs.append((score,cid))

# 按照分数降序排列（reverse=True 表示从大到小）
#lambda x: x[0] 等同于下面这个函数：
#def 拿第一个元素(x):
#    return x[0]
score_id_pairs.sort(key=lambda x:x[0],reverse=True)

# 打印出分数最高的前 5 个
print("\n--- BM25 检索结果 Top 5 ---")
top_5=score_id_pairs[:5]
for rank,(score,cid) in enumerate(top_5,1):
    print(f"排名{rank}|分数:{score:.4f}|Chunk ID:{cid}")

# 7. 验证一下（看看这 Top5 里有没有你标注的 gold_chunk_id）
gold_id=test_questions[0]["gold_chunk_id"]
print(f"\n你的标准答案所在的Chunk:{gold_id}")

# ================= 新增：向量检索部分 =================
print("\n--- 开始计算向量检索 ---")
# 1. 加载一个轻量级的多语言开源模型（支持中文）
# 第一次运行会下载模型，如果卡住，去喝杯水
model=SentenceTransformer('./local_model')

# 2. 把所有的 chunk 文本变成向量（这步叫 Encoding）
corpus_embeddings=model.encode(corpus)

# 3. 把问题也变成向量
query_embedding=model.encode(query)

# 4. 计算问题和每个 chunk 向量的余弦相似度（越接近 1 越相似）
cos_scores=util.cos_sim(query_embedding,corpus_embeddings)[0]

# 5. 找 Top 5
# 同样的，把 (分数, ID) 绑在一起排序
dense_pairs=[]
for i in range(len(cos_scores)):
    dense_pairs.append((cos_scores[i].item(),chunk_ids[i]))

dense_pairs.sort(key=lambda x:x[0],reverse=True)
top_5_dense=dense_pairs[:5]

print("\n--- 向量检索结果 Top 5 ---")
for rank,(score,cid) in enumerate(top_5_dense,1):
    print(f"排名{rank}|相似度：{score:.4f}|Chunk ID:{cid}")

# ================= 新增：RRF 混合检索融合 =================
print("\n--- 开始 RRF 混合检索融合 ---")

# RRF 公式：score = 1 / (k + rank)。rank 是排名，k 一般取 60
k=60
rrf_scores={}

# 1. 把 BM25 的结果加进去（注意取前 10 名或者全部）
for rank,(score,cid) in enumerate(score_id_pairs,1):
    if cid not in rrf_scores:
        rrf_scores[cid]=0
    rrf_scores[cid]+=1/(k+rank)

# 2. 把 向量检索 的结果加进去
for rank,(score,cid) in enumerate(dense_pairs,1):
    if cid not in rrf_scores:
        rrf_scores[cid]=0
    rrf_scores[cid]+=1/(k+rank)

# 3. 按融合后的总分重新排序
final_ranking=sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)

print("\n--- 混合检索（RRF）最终结果 Top 5 ---")
for rank,(cid,score) in enumerate(final_ranking[:5],1):
    print(f"排名{rank}|RRF分数:{score:.5f}|Chunk ID:{cid}")

# ================= 新增：大模型生成（Generation） =================
print("\n--- 开始生成最终答案 ---")

# 1. 拿到 RRF 排序后的前 3 名（不需要全喂给大模型，太长了）
top_3_context=[]
for cid,score in final_ranking[:3]:#人工审查正确答案在第4第5，所以改成前5
    # 去你之前的 chunks 列表里，把对应的文本找出来
    for item in chunks:
        if item["id"]==cid:
            top_3_context.append(f"【{cid}】:{item['content']}")
            break

# for score,cid in score_id_pairs[:3]:#（对照组：BM25）
#     # 去你之前的 chunks 列表里，把对应的文本找出来
#     for item in chunks:
#         if item["id"]==cid:
#             top_3_context.append(f"【{cid}】:{item['content']}")
#             break

# for score,cid in dense_pairs[:3]:#（对照组：向量）
#     # 去你之前的 chunks 列表里，把对应的文本找出来
#     for item in chunks:
#         if item["id"]==cid:
#             top_3_context.append(f"【{cid}】:{item['content']}")
#             break

# 把 Top 3 拼成一个大字符串
context_text="\n\n".join(top_3_context)
print(f"喂给大模型的参考资料：\n{context_text}\n")

# 2. 组装 Prompt（提示词工程）
prompt=f"""你是一个严谨的问答助手。请仅根据一下提供的参考资料回答问题。
如果你在资料中找不到答案，请直接回答“根据已知资料无法回答”。
回答时，请保持简洁，并指出你的答案来源于哪个资料ID。

参考资料：
{context_text}

问题：{query}
"""

# 3. 调用你本地的 Ollama 模型（注意换成你本地跑起来的模型名字，大概率是 qwen2.5-coder:7b）
print("本地模型正在思考中...\n")
response=ollama.chat(model='qwen2.5-coder:7b',messages=[
    {
        'role':'user',
        'content':prompt,
    },
])

# 4. 打印大模型的最终回答
print("--- 大模型最终回答 ---")
print(response['message']['content'])