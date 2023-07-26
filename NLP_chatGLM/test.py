import pandas as pd
import copy
import zhipuai
import time
from tqdm import tqdm

train = pd.read_csv("train.csv")
testB = pd.read_csv("testB.csv")


zhipuai.api_key = "YOUR API KEY"

# 训练集的制作
tmp = []
for i in range(5):
    train_item = train.loc[i]
    text = train_item[1] + ' ' + train_item[2] + ' ' + train_item[3]
    instruction = {"role": "user",
                   "content": f"Please judge judge the key words according to the given paper title and abstract, and then provide the keywords, the following is the paper title, author and abstract -->{text}"}
    answer = {"role": "assistant", "content": train_item[4]}
    tmp.append(instruction)
    tmp.append(answer)

print(tmp)

# 预测函数
def predict(test_item, tmp):
    prompt = copy.deepcopy(tmp)
    test = test_item[1] + ' ' + test_item[2] + ' ' + test_item[3]
    test_instruction = {"role": "user",
                        "content": f"Please judge judge the key words according to the given paper title and abstract, and then provide the keywords, the following is the paper title, author and abstract -->{test}"}
    prompt.append(test_instruction)
    response = zhipuai.model_api.invoke(
        model="chatglm_std",
        prompt=prompt
    )
    print(response)
    data = response['data']
    choices = data['choices']
    choices = choices[0]
    content = choices['content']
    content=content.replace('"','')
    content=content.replace(' ','',1)
    return content


test_keywords=[]
# 测试集的制作
for i in tqdm(range(len(testB))):
    test_item = testB.loc[i]
    Keywords=predict(test_item, tmp)
    test_keywords.append(Keywords)
    time.sleep(0.5)

testB['label']=label['label']
testB['Keywords']=test_keywords

testB[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)


# train_item=train.loc[0]
# text=train_item[1]+' '+train_item[2]+' '+train_item[3]
# test_item=testB.loc[0]
# test=test_item[1]+' '+test_item[2]+' '+test_item[3]
# tmp=[
#     {"role":"user","content":f"Please judge whether it is a medical field paper according to the given paper title and abstract, output 1 or 0 and then provide the keywords, the following is the paper title, author and abstract -->{text}"},
#     {"role":"assistant","content":f"{train_item[5]},{train_item[4]}"},
#     {"role":"user","content":f"Please judge whether it is a medical field paper according to the given paper title and abstract, output 1 or 0 and then provide the keywords, the following is the paper title, author and abstract -->{test}"}
# ]
#
# response=zhipuai.model_api.invoke(
#     model="chatglm_lite",
#     prompt=tmp
# )


# response = zhipuai.model_api.invoke(
#     model="chatglm_lite",
#     prompt=[
#         {"role": "user", "content": "你好"},
#         {"role": "assistant", "content": "我是人工智能助手"},
#         {"role": "user", "content": "你叫什么名字"},
#         {"role": "assistant", "content": "我叫chatGLM"},
#         {"role": "user", "content": "你都可以做些什么事"},
#     ]
# )

# data = response['data']
# choices = data['choices']
# choices=choices[0]
# content=choices['content']
#
# print(content)
