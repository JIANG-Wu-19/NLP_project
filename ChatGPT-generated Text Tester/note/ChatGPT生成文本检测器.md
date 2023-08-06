# ChatGPT生成文本检测器

题目链接：[ChatGPT生成文本检测器](https://challenge.xfyun.cn/topic/info?type=text-detector&option=phb)

友情链接：[Datawhale-NLP实践](https://datawhaler.feishu.cn/docx/OQjWdEuAZo2nTSxHmT6crhZ2nmf)

项目地址：[NLP_project](https://github.com/JIANG-Wu-19/NLP_project)

## 跑通一个baseline

这次的baseline使用**手工确定提取文本特征**，在第四点中有体现

1. **导入库**：开始时导入必要的库，包括numpy、pandas和scikit-learn的LogisticRegression。

2. **加载数据**：加载两个CSV文件——'train.csv'和'test.csv'，并分别存储在pandas的DataFrame 'train_data'和'test_data'中。

3. **数据预处理**：对'train_data'和'test_data'的'content'列应用了一些预处理步骤。它使用lambda函数从'content'列中的每个字符串中移除第一个和最后一个字符，也就是移除“[”和“]”

4. **特征工程**：定义一个名为'simple_feature(s)'的函数，它接受一个字符串's'作为输入，并基于输入字符串的特性返回一个numpy数组作为特征，特征都是人为的选取所以效果并不是特别好这些特征包括：
   - 原始字符串的长度。
   - 字符串中字符的数量。
   - 字符串中唯一字符的数量。
   - 字符串中重复字符的数量。
   - 唯一字符数占总字符数的比例。
   - 字符串中字符频率的最大值。
   - 字符串中字符频率的最小值。
   - 字符串中字符频率的平均值。
   - 字符串中字符频率的标准差。
   - 字符串中字符频率的极差（最大值与最小值之间的差值）。

5. **生成训练和测试特征**：使用`apply`方法将'simple_feature'函数应用于'train_data'和'test_data'的'content'列中的每个字符串。得到的特征被垂直堆叠以创建numpy数组'train_feature'和'test_feature'。

6. **模型训练**：通过调用scikit-learn的`LogisticRegression()`来初始化一个逻辑回归模型'm'。然后，使用`fit`方法将训练特征'train_feature'和'train_data'中对应的'label'列用于训练模型。

7. **进行预测**：训练好的模型使用`predict`方法对测试数据进行预测，得到测试集的标签。

8. **将结果保存为CSV**：将预测的标签添加到'test_data'的'label'列中，然后创建一个只包含'name'和'label'列的新DataFrame。最后，将这个DataFrame保存为名为'simple.csv'的CSV文件，可以用于提交或进一步分析。

baseline跑出来的分数中规中矩，只有0.85+，还需要进行调整

详细见附录

## 冲榜尝试，upper的产生

upper基于TF-IDF，简单了解一下TF-IDF

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) 是一种在文本挖掘和信息检索中常用的特征表示方法，用于衡量一个词语对于一个文档集合中某个文档的重要程度。

TF-IDF 特征由两部分组成：Term Frequency (TF) 和 Inverse Document Frequency (IDF)。

1. **Term Frequency (TF)**：指定词语在文档中出现的频率。它表示一个词在文档中的重要程度。一般情况下，一个词在文档中出现的次数越多，它对文档的内容表达越重要。

   TF = (词语在文档中出现的次数) / (文档中总词语数)

2. **Inverse Document Frequency (IDF)**：指定词语在整个文档集合中的重要程度。它是通过对所有文档计算某个词的出现频率，然后取其倒数来计算的。IDF 的目的是降低在多个文档中频繁出现的常见词的权重，增加对于少数文档中出现但对于整个文档集合较为重要的词的权重。

   IDF = log(文档集合中的文档总数 / 含有该词的文档数 + 1)

最终的 TF-IDF 特征是将 TF 和 IDF 相乘得到的。TF-IDF 特征考虑了一个词在当前文档中的重要性（TF），以及它在整个文档集合中的全局重要性（IDF）。因此，TF-IDF 特征能够在文本分类和信息检索任务中更好地捕捉每个词在文档中的信息价值，从而在构建特征表示时更好地区分不同的文档。

### upper的思路

1. **导入库**：开始时导入所需的库，包括pandas、sklearn的LogisticRegression类、TfidfVectorizer类以及一些评估指标相关的库。

2. **加载数据**：加载两个CSV文件——'train.csv'和'test.csv'，并将它们分别存储在pandas的DataFrame 'train_data'和'test_data'中。

3. **数据预处理**：对'train_data'和'test_data'的'content'列应用了一些预处理步骤。它使用lambda函数从'content'列中的每个字符串中移除第一个和最后一个字符。

4. **TF-IDF特征提取**：
   - 使用TfidfVectorizer类来将文本内容转换为TF-IDF向量表示。
   - 尝试了三种不同的TF-IDF参数设置：
     - 第1种参数：token_pattern=r'\w{1}'，max_features=2000。token_pattern指定了用于提取词语的正则表达式，r'\w{1}'表示提取长度为1的单词。max_features设置为2000表示最多选择2000个最重要的特征词。
     - 第2种参数：token_pattern=r'\w{1}'，max_features=5000。max_features设置为5000表示最多选择5000个最重要的特征词。
     - 第3种参数：token_pattern=r'\w{1}'，max_features=5000，ngram_range=(1,2)。ngram_range=(1,2)表示同时提取单个词和二元（两个词）的组合作为特征。
   - 对于每种参数设置，分别将训练集和测试集的文本内容转换为TF-IDF向量表示。

5. **模型训练和评估**：
   - 对于每种TF-IDF参数设置，使用交叉验证（cross_val_predict）对逻辑回归模型进行评估，以得到更稳定的性能评估结果。
   - 使用`cross_val_predict`方法对训练集的TF-IDF向量和对应的标签进行交叉验证预测，然后通过classification_report函数输出分类结果的评估报告，包括精确度、召回率、F1-score等指标。

6. **模型训练和预测**：
   - 选择其中一种TF-IDF参数设置（这里使用第3种参数）来训练逻辑回归模型。
   - 使用训练好的模型对测试集的TF-IDF向量进行预测，并将预测结果保存在'test_data'的'label'列中。

7. **保存结果为CSV文件**：
   - 将测试集的'name'和'label'列保存为名为'tfidf.csv'的CSV文件，用于后续的提交或分析。

总体来说，使用TF-IDF特征提取方法和逻辑回归模型进行文本分类任务，并通过交叉验证来评估模型的性能。通过尝试不同的TF-IDF参数设置，可以找到最佳的特征提取策略，以获得更好的分类结果。

对于upper，冲榜的分数能够达到0.91+，但是这还不够，可以考虑之前使用过的BERT模型进行分类

详细代码见附录

## 附录

### baseline.py

```python
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

```

### upper.py

```python
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


```







> To Be Continued