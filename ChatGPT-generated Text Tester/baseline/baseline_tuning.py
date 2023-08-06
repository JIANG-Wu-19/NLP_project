import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

train_data = pd.read_csv('train.csv')
train_data['content'] = train_data['content'].apply(lambda x: x[1:-1])


# 定义参数搜索空间
param_dist = {
    'C': uniform(0.1, 10),       # LogisticRegression的惩罚参数C在0.1~10之间均匀分布
    'penalty': ['l1', 'l2'],     # 正则化类型：L1正则化或L2正则化
    'max_iter': [1000, 2000, 3000]  # 最大迭代次数
}


def simple_feature(s):
    if len(s) == 0:
        s = '123 123'

    w = s.split()

    # 统计字符出现次数
    w_count = np.bincount(w)
    w_count = w_count[w_count != 0]

    return np.array([

        len(s),  # 原始字符长度
        len(w),  # 字符个数
        len(set(w)),  # 不重复字符个数
        len(w) - len(set(w)),  # 字符个数 - 不重复字符个数
        len(set(w)) / (len(w) + 1),  # 不重复字符个数占比

        np.max(w_count),  # 字符的频率的最大值
        np.min(w_count),  # 字符的频率的最小值
        np.mean(w_count),  # 字符的频率的平均值
        np.std(w_count),  # 字符的频率的方差
        np.ptp(w_count),  # 字符的频率的极差
    ])


train_feature = train_data['content'].iloc[:].apply(simple_feature)
train_feature = np.vstack(train_feature.values)

# 创建LogisticRegression分类器
logreg = LogisticRegression()

# 初始化RandomizedSearchCV，传入模型、参数搜索空间、交叉验证的折数等
random_search = RandomizedSearchCV(estimator=logreg, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=10)

# 开始参数搜索与交叉验证
random_search.fit(train_feature, train_data['label'])

# 输出最佳参数组合
print("Best parameters: ", random_search.best_params_)
