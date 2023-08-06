from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

import pandas as pd

# 读取数据集
train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')


# 定义参数搜索空间
param_dict = {
    'C': uniform(0.1, 10),       # LogisticRegression的惩罚参数C在0.1~10之间均匀分布
    'penalty': ['l1', 'l2'],     # 正则化类型：L1正则化或L2正则化
    'max_iter': [1000, 2000, 3000]  # 最大迭代次数
}

# 创建CountVectorizer
vectorizer = CountVectorizer()

# 转换文本数据为特征向量
train_vector = vectorizer.fit_transform(train['text'])

# 创建LogisticRegression分类器
logreg = LogisticRegression()

# 初始化RandomizedSearchCV，传入模型、参数搜索空间、交叉验证的折数等
random_search = RandomizedSearchCV(estimator=logreg, param_distributions=param_dict, cv=5, n_jobs=-1, n_iter=10)

# 开始参数搜索与交叉验证
random_search.fit(train_vector, train['label'])

# 输出最佳参数组合
print("Best parameters: ", random_search.best_params_)
