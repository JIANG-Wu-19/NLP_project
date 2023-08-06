import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import ErnieModel
from transformers import AutoModelForSequenceClassification

data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

model=AutoModelForSequenceClassification.from_pretrained("ernie-3.0-mini-zh",num_classes=2)
tokenizer=AutoTokenizer.from_pretrained("ernie-3.0-mini-zh")

# 定义AdamW优化器，学习率为0.000001
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)

# 定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss(reduction='mean')






