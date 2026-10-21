import os
import json
import re

#打开data里准备好的数据文件并读取
with open("data/raw_text.txt","r",encoding="utf-8") as f:
    data=f.read()

# ================= 新增：数据清洗 =================
# 用正则表达式，把所有 [数字] 或 [数字-数字] 替换成空字符串
data=re.sub(r'\[\d+(-\d+)?\]','',data)

#对读取的内容进行切分并把符合要求的存入chunk
rechunks=data.split('\n')
chunks=[]#准备一个空列表
count=1#计数器
for chunk in rechunks:
    # 1. 新增：剥除两端的空格，如果是空行直接跳过！
    chunk=chunk.strip()
    if not chunk:
        continue
    
    # 2. 如果长度 <= 150，直接存
    if len(chunk)<=150:
        new_chunks={"id":f"chunk_{count}","content":chunk}#使列表的形式是chunk_number
        chunks.append(new_chunks)
        count+=1
    else:
        # 3. 把正则里的那个空格删掉！并加上中文的问号、感叹号
        tem_chunks=re.split(r'(?<=[。；.;！!？?])',chunk)

        # 4. 用一个临时碗来装句子（优化切分太碎的问题）
        current_chunk=""
        for chunk2d in tem_chunks:
            # 如果加上这句话还没超过 150 字，就拼在一起
            if len(current_chunk)+len(chunk2d)<=150:
                current_chunk+=chunk2d
            else:
                # 装不下了，就把手里拼好的存进 chunks
                if current_chunk:
                    new_chunks={"id":f"chunk_{count}","content":current_chunk}
                    chunks.append(new_chunks)
                    count+=1
                # 然后把这句新的作为新碗的开头
                current_chunk=chunk2d

        # 循环结束后，把碗里最后剩下的一点存进去
        if current_chunk:
            new_chunks={"id":f"chunk_{count}","content":current_chunk}#使列表的形式是chunk_number
            chunks.append(new_chunks)
            count+=1

#把 chunks 列表用 json.dump() 存到 data/chunks.json 文件里
with open("data/chunks.json","w",encoding="utf-8") as f:
    json.dump(chunks,f,ensure_ascii=False,indent=2)