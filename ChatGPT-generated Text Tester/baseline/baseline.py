import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['content'] = train_data['content'].apply(lambda x: x[1:-1])
test_data['content'] = test_data['content'].apply(lambda x: x[1:-1])


# Get the feature
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
test_feature = test_data['content'].iloc[:].apply(simple_feature)

train_feature = np.vstack(train_feature.values)
test_feature = np.vstack(test_feature.values)

# 模型训练
m = LogisticRegression()
m.fit(train_feature, train_data['label'])

# 生成测试集提交结果
test_data['label'] = m.predict(test_feature)
test_data[['name', 'label']].to_csv('simple.csv', index=None)
