import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['content'] = train_data['content'].apply(lambda x: x[1:-1])
test_data['content'] = test_data['content'].apply(lambda x: x[1:-1])

# 第1种tfidf参数
tfidf = TfidfVectorizer(token_pattern=r'\w{1}', max_features=2000)
train_tfidf = tfidf.fit_transform(train_data['content'])
test_tfidf = tfidf.fit_transform(test_data['content'])
print(classification_report(
    cross_val_predict(
        LogisticRegression(),
        train_tfidf,
        train_data['label']
    ),
    train_data['label'],
    digits=4
))

# 第2种tfidf参数
tfidf = TfidfVectorizer(token_pattern=r'\w{1}', max_features=5000)
train_tfidf = tfidf.fit_transform(train_data['content'])
test_tfidf = tfidf.fit_transform(test_data['content'])
print(classification_report(
    cross_val_predict(
        LogisticRegression(),
        train_tfidf,
        train_data['label']
    ),
    train_data['label'],
    digits=4
))

# 第3种tfidf参数
tfidf = TfidfVectorizer(token_pattern=r'\w{1}', max_features=5000, ngram_range=(1,2))
train_tfidf = tfidf.fit_transform(train_data['content'])
test_tfidf = tfidf.fit_transform(test_data['content'])
print(classification_report(
    cross_val_predict(
        LogisticRegression(),
        train_tfidf,
        train_data['label']
    ),
    train_data['label'],
    digits=4
))


m = LogisticRegression()
m.fit(
    train_tfidf,
    train_data['label']
)

test_data['label'] = m.predict(test_tfidf)
test_data[['name', 'label']].to_csv('tfidf.csv', index=None)

