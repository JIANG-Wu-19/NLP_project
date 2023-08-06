from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pandas as pd
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

# 定义参数搜索空间
param_dist = {
    'n_estimators': randint(50, 200),       # 随机森林中树的数量在50~200范围内随机选择
    'max_depth': [None] + list(range(5, 30)),  # 树的最大深度在5~30范围内随机选择，None表示不限制最大深度
    'min_samples_split': randint(2, 20)     # 节点划分所需的最小样本数在2~20范围内随机选择
}

train = pd.read_csv('train.csv')
train['title'] = train['title'].fillna('')
train['abstract'] = train['abstract'].fillna('')

train['text'] = train['title'].fillna('') + ' ' +  train['author'].fillna('') + ' ' + train['abstract'].fillna('')+ ' ' + train['Keywords'].fillna('')

vector = CountVectorizer().fit(train['text'])
train_vector = vector.transform(train['text'])


# 创建随机森林分类器
rf = RandomForestClassifier()

# 初始化RandomizedSearchCV，传入模型、参数搜索空间、交叉验证的折数等
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=10)

# 开始参数搜索与交叉验证
random_search.fit(train_vector, train['label'])

# 输出最佳参数组合
print("Best parameters: ", random_search.best_params_)
