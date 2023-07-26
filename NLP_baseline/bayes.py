import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 数据预处理
data = pd.read_csv("your_dataset.csv")  # 替换为您的CSV文件路径
data = data.dropna()  # 去除缺失值
X = data["标题"] + " " + data["作者"] + " " + data["摘要"]  # 标题、作者和摘要组合为特征
y = data["医学领域标签"]  # 医学领域标签列，1表示医学领域，0表示非医学领域

# 文本预处理
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # 分词并转为小写
    tokens = [t for t in tokens if t.isalpha()]  # 去除非字母字符
    tokens = [t for t in tokens if t not in stop_words]  # 去除停用词
    return " ".join(tokens)

X = X.apply(preprocess_text)  # 对文本进行预处理

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征提取
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 模型训练
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# 模型测试
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

# 模型评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 关键词提取
# 对于医学领域的论文，可以使用TF-IDF或TextRank等算法提取关键词
# 对于非医学领域的论文，可以设置关键词为空值或默认值
# 请注意：关键词提取过程需要针对具体数据和任务选择合适的方法，并根据实际情况调整参数
