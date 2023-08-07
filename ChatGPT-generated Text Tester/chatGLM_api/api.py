import pandas as pd
import copy
import zhipuai
import time
from tqdm import tqdm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['content'] = train['content'].apply(lambda x: x[1:-1])
test['content'] = test['content'].apply(lambda x: x[1:-1])

zhipuai.api_key = "YOUR_API_KEY"

n = 5
prompt_data = train.sample(n=n)
prompt_data = prompt_data.reset_index(drop=True)

print(prompt_data)

tmp = []
for i in range(len(prompt_data)):
    item = prompt_data.loc[i]
    instruction = {
        "role": "user",
        "content": f"将给定的文本分成两类，并给出标签0或1，给定文本如下：{item[2]}"
    }
    answer = {
        "role": "assistant",
        "content": f"{item[1]}"
    }
    tmp.append(instruction)
    tmp.append(answer)

print(tmp)

test_item = test.loc[0]


def predict(test_item, tmp):
    prompt = copy.deepcopy(tmp)
    test_instruction = {
        "role": "user",
        "content": f"将给定的文本分成两类，并给出标签0或1，给定文本如下：{test_item[1]}"
    }
    prompt.append(test_instruction)
    print(prompt)
    # response = zhipuai.model_api.invoke(
    #     model="chatglm_pro",
    #     prompt=prompt
    # )
    # print(response)


predict(test_item=test_item, tmp=tmp)
