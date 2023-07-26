# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

# 导入LogisticRegression回归模型
from sklearn.linear_model import LogisticRegression

# 过滤警告消息
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# 引入分词器
from nltk import word_tokenize, ngrams
simplefilter("ignore", category=ConvergenceWarning)

from nltk.corpus import stopwords


param_dist = {
    'C': 2.8496190796154743,       # LogisticRegression的惩罚参数C在0.1~10之间均匀分布
    'penalty': 'l2',     # 正则化类型：L1正则化或L2正则化
    'max_iter': 3000  # 最大迭代次数
}


# 读取数据集
train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('testB.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])


# 引入模型
model = LogisticRegression(**param_dist)

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'label']].to_csv('submit_task1.csv', index=None)


# 定义停用词，去掉出现较多，但对文章不关键的词语
stops = set(stopwords.words())


# 定义方法按照词频筛选关键词

def extract_keywords_by_freq(title, abstract):
    ngrams_count = list(ngrams(word_tokenize(title.lower()), 2)) + list(ngrams(word_tokenize(abstract.lower()), 2))
    ngrams_count = pd.DataFrame(ngrams_count)
    ngrams_count = ngrams_count[~ngrams_count[0].isin(stops)]
    ngrams_count = ngrams_count[~ngrams_count[1].isin(stops)]
    ngrams_count = ngrams_count[ngrams_count[0].apply(len) > 3]
    ngrams_count = ngrams_count[ngrams_count[1].apply(len) > 3]
    ngrams_count['phrase'] = ngrams_count[0] + ' ' + ngrams_count[1]
    ngrams_count = ngrams_count['phrase'].value_counts()
    ngrams_count = ngrams_count[ngrams_count > 1]
    return list(ngrams_count.index)[:6]


## 对测试集提取关键词

test_words = []
for row in test.iterrows():
    # 读取第每一行数据的标题与摘要并提取关键词
    prediction_keywords = extract_keywords_by_freq(row[1].title, row[1].abstract)
    # 利用文章标题进一步提取关键词
    prediction_keywords = [x.title() for x in prediction_keywords]
    # 如果未能提取到关键词
    if len(prediction_keywords) == 0:
        prediction_keywords = ['A', 'B']
    test_words.append('; '.join(prediction_keywords))

test['Keywords'] = test_words
test[['uuid', 'Keywords', 'label']].to_csv('submit_task2.csv', index=None)