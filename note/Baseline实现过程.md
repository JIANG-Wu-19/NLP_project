# Baseline实现过程

## import库

```python
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
```

其中比较重要的几个库：`pandas`,`sklearn`,`nltk`。上述三个库进行分开讲解。

### pandas

* pandas是一个第三方的库，因此需要先下载

```
pip install pandas
```

pandas的功能非常强大，早在学习的时候便能窥见一二。

pandas的主要数据结构`Series`和`DataFrame`，分别对应一维数据和二维数据

Pandas 就像一把万能瑞士军刀，下面仅列出了它的部分优势 ：

- 处理浮点与非浮点数据里的**缺失数据**，表示为 `NaN`；
- 大小可变：**插入或删除** DataFrame 等多维对象的列；
- 自动、显式**数据对齐**：显式地将对象与一组标签对齐，也可以忽略标签，在 Series、DataFrame 计算时自动与数据对齐；
- 强大、灵活的**分组**（group by）功能：**拆分-应用-组合**数据集，聚合、转换数据；
- 把 Python 和 NumPy 数据结构里不规则、不同索引的数据**轻松**地转换为 DataFrame 对象；
- 基于智能标签，对大型数据集进行**切片**、**花式索引**、**子集分解**等操作；
- 直观地**合并（merge）**、**连接（join）**数据集；
- 灵活地**重塑（reshape）**、**透视（pivot）**数据集；
- **轴**支持结构化标签：一个刻度支持多个标签；
- 成熟的 IO 工具：读取**文本文件**（CSV 等支持分隔符的文件）、Excel 文件、数据库等来源的数据，利用超快的 **HDF5** 格式保存 / 加载数据；
- **时间序列**：支持日期范围生成、频率转换、移动窗口统计、移动窗口线性回归、日期位移等时间序列功能。

在这里，我们主要用到的就是读取csv文件，然后对读取到的数据进行行操作、列操作。

### sklearn

scikit-learn，又写作sklearn，是一个开源的基于python语言的机器学习工具包。它通过NumPy, SciPy和Matplotlib等python数值计算的库实现高效的算法应用，并且涵盖了几乎所有主流机器学习算法。

在这里baseline使用的是Logistic Regression，当然对于分类而言，还可以使用random forest、SVM等等，也是sklearn库中的模块，可以在baseline的基础上进行操作。

### nltk

nltk全称Natural Language Toolkit，自然语言处理工具包，在NLP领域中，最常使用的一个Python库。

在baseline中，使用了nltk库，对文本进行一系列操作如单词切分、去除停用词、抽取关键词等等操作

## 实现任务一

* task 1：判断是否是医学论文

官方给出的A集中已经含有Keywords一项，也就是说在前期工作中并不需要完成任务二。

对于任务一来说，是一个简单的文本分类问题，官方给出了训练集，那么可以采用Logistic Regression来做分类的预测，实际上random forest和SVM也可以，在后文会略微提到一点，因为没有得到显著的提升效果。

### Logistic Regression

逻辑回归（Logistic Regression）与线性回归（Linear Regression）都是一种广义线性模型（generalized linear model）。逻辑回归假设因变量 y 服从伯努利分布，而线性回归假设因变量 y 服从高斯分布。 因此与线性回归有很多相同之处，去除Sigmoid映射函数的话，逻辑回归算法就是一个线性回归。可以说，逻辑回归是以线性回归为理论支持的，但是逻辑回归通过Sigmoid函数引入了非线性因素，因此可以轻松处理0/1分类问题。

首先引入Sigmoid函数，称逻辑函数：
$$
g(z)= \frac {1}{1+e^{-z}}
$$
上述函数对于解决二分类问题十分重要

逻辑回归的假设函数形式如下：
$$
h_\theta(x)=g\left(\theta^T x\right), g(z)=\frac{1}{1+e^{-z}}
$$
得到：
$$
h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}
$$
$x$是输入数据，$\theta$是求取参数

逻辑回归就是计算在给定上述两个参数的条件下$y=1$的概率。

做假设：
$$
P(y=1 \mid x ; \theta)=g\left(\theta^T x\right)=\frac{1}{1+e^{-\theta^T}}
$$
决策函数就是：
$$
P(y=1|x)>0.5 \Rightarrow y^*=1
$$

### 代码实现

上一节中都是Logistic Regression的原理，在这里一笔带过，如果单纯的手撕逻辑回归代码会相当耗时，sklearn库已经为我们封装好了模型，只要将训练集的相关列组成文本向量然后进行训练就可以得到预测结果。

首先使用pandas提供的IO操作进行数据集的读取：

```python
# 读取数据集
train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

test = pd.read_csv('test.csv')
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')
```

然后进行训练集、测试集的文本拼接：

```python
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
```

以上都是数据的预处理，接下来才是重点

* 将文本向量化

  ```python
  vector = CountVectorizer().fit(train['text'])
  train_vector = vector.transform(train['text'])
  test_vector = vector.transform(test['text'])
  ```

* 将数据集输入模型进行训练

  ```python
  # 引入模型
  model = LogisticRegression(**param_dist)
  
  # 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
  model.fit(train_vector, train['label'])
  ```

  其中，Logistic Regression模型中的超参数在另一个程序中通过交叉验证得到，以下是程序运行后得到的一个优解

  ```python
  param_dict = {
      'C': 2.8496190796154743,       # LogisticRegression的惩罚参数C在0.1~10之间均匀分布
      'penalty': 'l2',     # 正则化类型：L1正则化或L2正则化
      'max_iter': 3000  # 最大迭代次数
  }
  ```

  具体程序不在这里赘述，感兴趣的见附录。

* 对测试机label进行预测

  ```python
  test['label'] = model.predict(test_vector)
  ```

* 生成csv文件

  ```python
  test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)
  ```

最终，在笔者使用了一点trick的情况下，跑出来的分数还是比较可观的

![image-20230727173735393](img\2.png)

但是，如果是关键词未给出的B集，Logistic Regression的表现就会比较糟糕

## 任务一模型的选择

在超参数缺省的情况下，random forest的表现最好，随后依次是Logistic Regression、SVM；

如果分别对模型的参数进行交叉验证得到一组优化的参数，Logistic Regression的表现最好，随后依次是random forest、SVM；

也就是说，SVM在本分类中并不适合。

Logistic Regression和random forest两者的差距不大，都可以选择

## 实现任务二

* task 2：关键词抽取

官方更新的B集中，去除了Keywords一列，因此需要进行提取

baseline中的实现过程是对文本进行关键词的抽取

相关库已经import，在上述任务一的基础上进行后续操作



* 停用词的设置，这里直接使用nltk自带的停用词库

  ```python
  stops = set(stopwords.words())
  ```

* 定义筛选函数

  从给定文档的标题和摘要中提取关键词吗，遵循特定的过程来识别频繁出现的词组（二元组），作为潜在的关键词。

  ```python
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
  ```

  * `ngrams_count = list(ngrams(word_tokenize(title.lower()), 2)) + list(ngrams(word_tokenize(abstract.lower()), 2))`：首先，将`title`和`abstract`拆分成小写单词，并从标题和摘要中创建二元组（两个连续的单词序列）。这些二元组存储在`ngrams_count`列表中
  * `ngrams_count = pd.DataFrame(ngrams_count)`：将`ngrams_count`列表中的二元组转换为pandas DataFrame，以便更容易地进行数据处理
  * `ngrams_count = ngrams_count[~ngrams_count[0].isin(stops)]` 和 `ngrams_count = ngrams_count[~ngrams_count[1].isin(stops)]`：在这里，函数过滤掉包含停用词的二元组。停用词是常用词，在文本分析中通常被忽略，因为它们没有很重要的意义
  * `ngrams_count = ngrams_count[ngrams_count[0].apply(len) > 3]` 和 `ngrams_count = ngrams_count[ngrams_count[1].apply(len) > 3]`：函数过滤掉第一个和第二个单词长度小于等于3个字符的二元组。这样做是为了去除非常短和不太有信息量的短语
  * `ngrams_count['phrase'] = ngrams_count[0] + ' ' + ngrams_count[1]`：将每个二元组的第一个单词和第二个单词连接起来，创建一个名为'phrase'的新列
  * `ngrams_count = ngrams_count['phrase'].value_counts()`：然后，函数计算每个唯一二元组短语的出现次数
  * `ngrams_count = ngrams_count[ngrams_count > 1]`：它过滤掉仅出现一次的二元组短语，因为它们可能作为关键词的相关性较低
  * `return list(ngrams_count.index)[:6]`：最后，函数返回出现频率最高的前6个二元组短语作为提取的关键词

* 对测试集提取关键词

  使用之前定义的`extract_keywords_by_freq`函数从给定的测试数据中提取关键词，并将提取到的关键词拼接成一个字符串

  ```python
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
  ```

  * `prediction_keywords = extract_keywords_by_freq(row[1].title, row[1].abstract)`: 从当前行的标题和摘要中调用`extract_keywords_by_freq`函数提取关键词，并将结果存储在`prediction_keywords`变量中。

  * `prediction_keywords = [x.title() for x in prediction_keywords]`: 将`prediction_keywords`中的每个关键词的首字母大写，形成一个新的列表，这是为了统一关键词的格式。

  * `if len(prediction_keywords) == 0: prediction_keywords = ['A', 'B']`: 如果未能提取到关键词（即`prediction_keywords`为空列表），则将默认关键词`['A', 'B']`赋值给`prediction_keywords`。

  * `test_words.append('; '.join(prediction_keywords))`: 将处理后的关键词列表转换为以分号为分隔符的字符串，并将这个字符串添加到`test_words`列表中。

* 将得到的test_words存入Series中，并生成csv文件

  ```python
  test['Keywords'] = test_words
  test[['uuid', 'Keywords', 'label']].to_csv('submit_task2.csv', index=None)
  ```

抽取式关键字存在比较大的缺陷，文献的标题、摘要中不一定会出现关键字，因此造成的准确率不高，跑出来的分数也不是很好看。

## 任务顺序的变换

由于任务二是提取关键字，而任务一的准确率又和关键字挂钩，有没有可能通过调整任务的顺序，先提取关键字，然后将关键字加入进行训练以得到更准确的分类。

这种idea很容易实现，因为代码都是现成的，只需要稍加改动就可以了，原理在前面两节都已经提及，实现代码见附录。

虽然分数都不是很好看，但是有那么一丢丢进步。

![image-20230727173918592](img\3.png)

![image-20230727174658234](img\4.png)

## 不足

上述都只是一个baseline的实现过程，也就是说都还有进步空间，对于传统机器学习，已经到达极限。

后续笔者冲榜的时候对任务一的实现进行改进，使用了BERT模型，对于文本分类的准确度，有较大的提升。

对于任务二，笔者没有进行改动，但是看了群友的思路后，觉得抽取式关键词的准确率也就那样了，可能要转向生成式关键词，也就是使用LLM，笔者困于时间，在deadline前没有足够时间完成

## 附录

### baseline_tuning.py

```python
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
param_dist = {
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
random_search = RandomizedSearchCV(estimator=logreg, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=10)

# 开始参数搜索与交叉验证
random_search.fit(train_vector, train['label'])

# 输出最佳参数组合
print("Best parameters: ", random_search.best_params_)

```

### baseline1.py

```python
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

param_dist = {
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
model = LogisticRegression(**param_dist)

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('submit_task1.csv', index=None)
```

### baseline2.py

```python
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
```

### baseline3.py

```python
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
# fillna()函数用于填充缺失数据，这里将标题和摘要的缺失值填充为空字符串
test['title'] = test['title'].fillna('')
test['abstract'] = test['abstract'].fillna('')


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])

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


# 对测试集提取关键词

test_words = []
for row in test.iterrows():
    # 读取第每一行数据的标题与摘要并提取关键词
    prediction_keywords = extract_keywords_by_freq(row[1].title, row[1].abstract)
    # 利用文章标题进一步提取关键词
    prediction_keywords = [x.title() for x in prediction_keywords]
    print(prediction_keywords)
    # 如果未能提取到关键词
    if len(prediction_keywords) == 0:
        prediction_keywords = ['A', 'B']
    test_words.append('; '.join(prediction_keywords))

test['Keywords'] = test_words
print(type(test['Keywords']))
print(test['Keywords'])

test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+' '+test['Keywords']
test_vector = vector.transform(test['text'])


# 引入模型
model = LogisticRegression(**param_dist)

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'label']].to_csv('submit_task1.csv', index=None)



test[['uuid', 'Keywords', 'label']].to_csv('submit_task2.csv', index=None)
```

### svm_test.py

```python
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

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


# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' +  test['author'].fillna('') + ' ' + test['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])


# 引入模型
model = SVC()

# 开始训练，这里可以考虑修改默认的batch_size与epoch来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('svm.csv', index=None)
```

### randomforest_test.py

```python
# 导入pandas用于读取表格数据
import pandas as pd

# 导入BOW（词袋模型），可以选择将CountVectorizer替换为TfidfVectorizer（TF-IDF（词频-逆文档频率）），注意上下文要同时修改，亲测后者效果更佳
from sklearn.feature_extraction.text import CountVectorizer

# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier

# 导入warnings库，过滤警告消息
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

# 提取文本特征，生成训练集与测试集
train['text'] = train['title'].fillna('') + ' ' + train['author'].fillna('') + ' ' + train['abstract'].fillna(
    '') + ' ' + train['Keywords'].fillna('')
test['text'] = test['title'].fillna('') + ' ' + test['author'].fillna('') + ' ' + test['abstract'].fillna('') + ' ' + \
               train['Keywords'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])
test_vector = vector.transform(test['text'])

# 引入随机森林分类器
model = RandomForestClassifier()

# 开始训练，可以考虑修改n_estimators（树的数量）和其他超参数来取得更好的效果
model.fit(train_vector, train['label'])

# 利用模型对测试集label标签进行预测
test['label'] = model.predict(test_vector)

# 生成任务一推测结果
test[['uuid', 'Keywords', 'label']].to_csv('randomforest.csv', index=None)

```

