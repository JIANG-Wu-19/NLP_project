# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# 读取数据集
train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('test.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')

param_dict = {
    'C': 2.8496190796154743,       # LogisticRegression的惩罚参数C在0.1~10之间均匀分布
    'penalty': 'l2',     # 正则化类型：L1正则化或L2正则化
    'max_iter': 3000  # 最大迭代次数
}


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])


# 引入模型
model = LogisticRegression(**param_dict)

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)