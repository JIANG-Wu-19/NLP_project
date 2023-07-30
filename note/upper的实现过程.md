# upper的实现过程

NLP进阶冲榜使用了BERT模型进行文本分类和关键词提取

## BERT模型

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

BERT模型由Google,stands for Bidirectional Encoder Representations from
Transformers.

> BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al.

BERT模型架构是一个多层的Transformer Encoder堆叠，也就是为人们所熟知的《[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)》中提出的多层Transformer结构，抛弃了传统的RNN和CNN，通过Attention机制将任意位置的两个单词的距离转换成1。

具体BERT原理在此不再赘述，值得关注的是BERT模型的训练包含pre-training和fine-tuning两个阶段，在这里用到的是BERT预训练的模型，使用训练集对BERT模型进行微调。

## 导入前置依赖

```python
#导入前置依赖
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 用于加载bert模型的分词器
from transformers import AutoTokenizer
# 用于加载bert模型
from transformers import BertModel
from pathlib import Path
```

- `os`: 用于操作文件路径和目录。
- `pandas (pd)`: 用于数据处理，读取CSV文件和处理DataFrame。
- `torch`: PyTorch库，用于构建深度学习模型。
- `torch.nn`: PyTorch中的神经网络模块。
- `torch.utils.data`: PyTorch中的数据加载和处理工具。
- `transformers`: Hugging Face的transformers库，用于加载和使用BERT等预训练模型。
- `pathlib.Path`: 用于处理文件路径。
- `TfidfVectorizer`: sklearn库中的TF-IDF向量化器，用于文本特征提取。
- `SentenceTransformer`: 用于获取句子的嵌入表示。
- `cosine_similarity`: 计算余弦相似度的函数。
- `nltk.word_tokenize`: 用于对文本进行分词。
- `nltk.corpus.stopwords`: NLTK中包含的英文停用词集合，用于文本预处理。

## 定义训练参数

```python
batch_size = 10
# 文本的最大长度
text_max_length = 128
# 总训练的epochs数，我只是随便定义了个数
epochs = 10
# 学习率
lr = 3e-5
# 取多少训练集的数据作为验证集
validation_ratio = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 每多少步，打印一次loss
log_per_step = 50

# 数据集所在位置
dataset_dir = Path("./基于论文摘要的文本分类与关键词抽取挑战赛公开数据")
os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else ''

# 模型存储路径
model_dir = Path("./model/bert_checkpoints")
# 如果模型目录不存在，则创建一个
os.makedirs(model_dir) if not os.path.exists(model_dir) else ''

print("Device:", device)
```

定义了各种常量，如批量大小(`batch_size`)、文本最大长度(`text_max_length`)、学习率(`lr`)、训练轮数(`epochs`)等。它还设定了设备类型(`device`)，如果有GPU，则设为'cuda'，否则设为'cpu'。

`log_per_step`表示每经过多少个步骤（batch）打印一次损失。

`dataset_dir`和`model_dir`分别定义了数据集和模型存储路径。`os.makedirs`用于创建这些目录。

在这里由于使用了anaconda的虚拟环境，也就是之前安装好的pytorch环境，内部已经配置好pytorch、transformer等，而且硬件方面有独显，所以device是cuda。

## 读取数据

```python
pd_train_data = pd.read_csv('train.csv')
pd_train_data['title'] = pd_train_data['title'].fillna('')
pd_train_data['abstract'] = pd_train_data['abstract'].fillna('')
test_data = pd.read_csv('testB.csv')
test_data['title'] = test_data['title'].fillna('')
test_data['abstract'] = test_data['abstract'].fillna('')
pd_train_data['text'] = pd_train_data['title'].fillna('') + ' ' + pd_train_data['author'].fillna('') + ' ' + \
                        pd_train_data['abstract'].fillna('') + ' ' + pd_train_data['Keywords'].fillna('')
test_data['text'] = test_data['title'].fillna('') + ' ' + test_data['author'].fillna('') + ' ' + test_data[
    'abstract'].fillna('') + ' ' + pd_train_data['Keywords'].fillna('')

```

使用`pandas`库从两个CSV文件（`train.csv`和`testB.csv`）中读取数据。然后，使用`fillna`方法填充缺失的标题（'title'）和摘要（'abstract'）字段，并将它们拼接成新的文本内容（'text'）。

然后从训练集中随机抽样，sample to train

```python
# 从训练集中随机采样测试集
validation_data = pd_train_data.sample(frac=validation_ratio)
train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]
```

## 构建自定义数据集类

```python
class MyDataset(Dataset):

    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.mode = mode
        # 拿到对应的数据
        if mode == 'train':
            self.dataset = train_data
        elif mode == 'validation':
            self.dataset = validation_data
        elif mode == 'test':
            # 如果是测试模式，则返回内容和uuid。拿uuid做target主要是方便后面写入结果。
            self.dataset = test_data
        else:
            raise Exception("Unknown mode {}".format(mode))

    def __getitem__(self, index):
        # 取第index条
        data = self.dataset.iloc[index]
        # 取其内容
        text = data['text']
        # 根据状态返回内容
        if self.mode == 'test':
            # 如果是test，将uuid做为target
            label = data['uuid']
        else:
            label = data['label']
        # 返回内容和label
        return text, label

    def __len__(self):
        return len(self.dataset)
```

自定义数据集类 `MyDataset` 的定义。在深度学习中，通常需要将数据封装成一个可迭代的数据集，以便于模型的训练和验证。PyTorch提供了`Dataset`和`DataLoader`这两个类来实现数据集的处理和加载。

`upper`中定义了一个自定义的数据集类 `MyDataset`，继承自PyTorch的`Dataset`类。这个类用于处理文本分类和关键词抽取的数据集，并根据传入的模式（'train'、'validation'、'test'）选择对应的数据子集。

在`MyDataset`类中，实现了以下几个重要的方法：

1. `__init__(self, mode='train')`: 这个方法是类的初始化方法，用于创建数据集对象时执行。`mode`参数用于指定数据集的模式，可以是'train'、'validation'或者'test'。根据不同的模式，选择对应的数据子集。
2. `__getitem__(self, index)`: 这个方法是数据集类的核心方法之一，用于返回指定索引`index`处的数据样本。在这里，根据索引取出对应的文本内容和标签。如果是测试模式，则将uuid作为标签，否则将标签取出。
3. `__len__(self)`: 这个方法返回数据集的大小，即样本的总数。

数据集类 `MyDataset` 的主要功能是根据不同的模式选择对应的数据子集，并根据索引取出对应的数据样本。当使用`DataLoader`加载数据时，它会调用`__getitem__`方法来获取每个批次的数据样本，并调用`__len__`方法获取数据集的大小。这样，就可以通过`DataLoader`来迭代地遍历整个数据集，方便地用于模型的训练和验证。

## 获取BERT预训练模型的tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

使用Hugging Face的`AutoTokenizer`类从预训练的BERT模型（'bert-base-uncased'）中获取对应的分词器。这个分词器将原始文本内容转换成BERT模型所需的输入格式

## 函数加载器的定义和使用

在深度学习中，数据集通常包含大量的样本，每个样本都有其对应的特征和标签。数据加载器是一个迭代器，它能够按照指定的批量大小（`batch_size`）将数据集划分成小批量进行加载和处理。

通常使用PyTorch的`DataLoader`来加载训练数据集和验证数据集，从而方便地将数据输入到模型中进行训练和验证。

### 数据加载器的定义

```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
```

* `train_dataset` 和 `validation_dataset`: 这两个参数分别是之前定义的自定义数据集类 `MyDataset` 的实例，用于加载训练数据集和验证数据集。

* `batch_size`: 这个参数指定每个批次的样本数。数据加载器会将数据集按照这个大小进行划分，每次返回一个小批量的样本。

* `shuffle`: 这个参数用于控制数据是否在每个epoch（训练周期）开始时随机打乱顺序。在训练时，通常会将数据打乱，以增加数据的随机性，有助于模型学习到更广泛的特征。但在验证时，不需要打乱数据，以保持数据的顺序。

* `collate_fn`: 这个参数用于定义如何将样本组合成一个批次。在这里，使用了之前定义的`collate_fn`函数。这个函数的作用是将每个样本中的文本数据（`text`）进行编码、填充，并组装成一个批次。具体地，它将样本的文本输入传递给BERT模型，得到BERT模型的输入，然后将输入和标签组装成一个字典形式的数据，用于模型的输入和训练。

其中的大部分参数都已经定义好了，比较重要的是`collate_fn`函数并没有定义

### collate_fn函数的定义

`collate_fn` 函数是在数据加载器（`DataLoader`）中使用的一个函数，用于对每个样本进行处理并组合成一个批次。在项目中，自定义了`collate_fn`函数，用于将每个样本中的文本数据进行编码、填充，并组装成一个批次，以便于模型的输入和训练。

```python
def collate_fn(batch):
    """
    将一个batch的文本句子转成tensor，并组成batch。
    :param batch: 一个batch的句子，例如: [('推文', target), ('推文', target), ...]
    :return: 处理后的结果，例如：
             src: {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]), 'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
             target：[1, 1, 0, ...]
    """
    text, label = zip(*batch)
    text, label = list(text), list(label)

    # src是要送给bert的，所以不需要特殊处理，直接用tokenizer的结果即可
    # padding='max_length' 不够长度的进行填充
    # truncation=True 长度过长的进行裁剪
    src = tokenizer(text, padding='max_length', max_length=text_max_length, return_tensors='pt', truncation=True)

    return src, torch.LongTensor(label)
```

1. 将文本列表转换为文本张量：使用 `tokenizer` 对文本进行编码，将文本转换为BERT模型需要的输入格式。`tokenizer` 是之前引入的一个来自 `transformers` 库的分词器（tokenizer），它可以将文本句子转换成BERT输入的 `input_ids` 和 `attention_mask` 张量。

2. padding和truncation：由于BERT模型的输入要求每个样本长度相同，因此需要对文本进行填充（padding）或截断（truncation）操作，使得所有文本都有相同的长度。在这里，将文本长度填充到 `text_max_length`，超过这个长度的部分会被截断。

3. 组装为字典形式：将编码后的文本张量 `input_ids` 和 `attention_mask` 组装成一个字典 `src`，它是模型的输入数据。`src` 的结构如下：

   ```
   {'input_ids': tensor([[ 101, ..., 102, 0, 0, ...], ...]),
    'attention_mask': tensor([[1, ..., 1, 0, ...], ...])}
   ```

   其中，`input_ids` 是文本的编码表示，`attention_mask` 是用于告诉BERT哪些部分是真实输入，哪些部分是填充的掩码。

4. 处理标签：将标签列表 `label` 转换为PyTorch的Long型张量 `torch.LongTensor(label)`。标签是用于文本分类的二分类问题，所以需要将标签转换成数值型数据。

最终，`collate_fn` 函数返回处理后的结果，包含了一个批次的输入数据 `src` 和对应的标签 `label`。这样，在使用 `DataLoader` 加载数据时，每次迭代返回的就是一个批次的数据，可以直接输入到模型中进行训练

## 定义预测模型

当训练深度学习模型时，需要定义一个模型结构，并确定用于评估模型性能的损失函数（loss function）。在这个代码中，定义了一个自定义的预测模型 `MyModel`，它由预训练的BERT模型和最后的预测层组成。在这个模型中，使用BCELoss作为损失函数，用于进行二分类任务的训练。

### 定义预测模型（MyModel）

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # 加载bert模型
        self.bert = BertModel.from_pretrained('bert-base-uncased', mirror='tuna')

        # 最后的预测层
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        """
        :param src: 分词后的推文数据
        """

        # 将src直接序列解包传入bert，因为bert和tokenizer是一套的，所以可以这么做。
        # 得到encoder的输出，用最前面[CLS]的输出作为最终线性层的输入
        outputs = self.bert(**src).last_hidden_state[:, 0, :]

        # 使用线性层来做最终的预测
        return self.predictor(outputs)
```

在`MyModel`中，首先定义了模型的初始化方法（`__init__`）。在这个方法中，做了以下几个事情：

1. 加载BERT模型：使用`BertModel.from_pretrained`方法加载了预训练的BERT模型。`'bert-base-uncased'`表示加载了小写字母的BERT模型，`mirror='tuna'`是为了使用TUNA（清华大学开源软件镜像站）的镜像下载预训练的权重。

2. 定义最后的预测层：通过`nn.Sequential`定义了一个包含多个线性层和激活函数的预测层。这个预测层用于将BERT模型的输出特征映射到一个单一的数值，用于进行二分类预测。在这里，使用了两个线性层，大小为768到256，再到1，以及一个ReLU激活函数和一个Sigmoid激活函数。

在神经网络中，最后的Sigmoid激活函数会将输出值压缩到0到1之间，用于表示概率。输出接近0表示负类（Negative Class），输出接近1表示正类（Positive Class）。的任务是根据文本内容对样本进行二分类，所以这个输出值可以表示样本属于正类的概率。

### forward()方法

在PyTorch中，每个自定义模型都需要定义`forward`方法。这个方法是模型的前向传递过程，用于定义数据从输入到输出的流动。在`MyModel`中，定义了`forward`方法，将输入的分词后的文本数据 `src` 传递给BERT模型，并通过最后的预测层获取模型的预测结果。

1. 将`src`输入BERT模型：将分词后的文本数据 `src` 传递给BERT模型。在这里，使用了双星号（`**`）将字典中的键值对拆分为参数形式，相当于`self.bert(input_ids=src['input_ids'], attention_mask=src['attention_mask'])`。`input_ids` 是分词后的句子的索引，`attention_mask` 是告诉BERT哪些部分是真实的输入，哪些部分是填充的。

2. 得到BERT输出：BERT模型的输出是一个包含多个隐藏层的元组，只需要最后一个隐藏层的输出。通过`last_hidden_state`选取最后一个隐藏层。

3. 使用线性层进行预测：最后，将BERT模型的输出传递给预测层 `self.predictor`，用于获得模型的预测结果。在这里，选取BERT输出中 `[CLS]` 符号对应的位置作为整个句子的表示，并通过预测层将其映射到一个单一的数值。

该方法的返回值就是模型的预测结果，即样本属于正类的概率。根据这个概率值，可以进行二分类预测。如果输出值接近0，则判定为负类；如果输出值接近1，则判定为正类。

总的来说，定义预测模型 `MyModel` 的过程是将预训练的BERT模型与一个预测层组合在一起，使得模型能够对输入文本进行二分类预测。预测层将BERT输出特征映射到一个数值，表示样本属

## 定义转移函数

定义了一个辅助函数`to_device`，用于将数据移动到指定的设备（CPU或GPU），以便在GPU上进行加速计算。

```python
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors
```

## 开始训练

```python
def validate():
    model.eval()
    total_loss = 0.
    total_correct = 0
    for inputs, targets in validation_loader:
        inputs, targets = to_device(inputs), targets.to(device)
        outputs = model(inputs)
        loss = criteria(outputs.view(-1), targets.float())
        total_loss += float(loss)

        correct_num = (((outputs >= 0.5).float() * 1).flatten() == targets).sum()
        total_correct += correct_num

    return total_correct / len(validation_dataset), total_loss / len(validation_dataset)


# 首先将模型调成训练模式
model.train()

# 清空一下cuda缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 定义几个变量，帮助打印loss
total_loss = 0.
# 记录步数
step = 0

# 记录在验证集上最好的准确率
best_accuracy = 0

# 开始训练
# for epoch in range(epochs):
#     model.train()
#     for i, (inputs, targets) in enumerate(train_loader):
#         # 从batch中拿到训练数据
#         inputs, targets = to_device(inputs), targets.to(device)
#         # 传入模型进行前向传递
#         outputs = model(inputs)
#         # 计算损失
#         loss = criteria(outputs.view(-1), targets.float())
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         total_loss += float(loss)
#         step += 1
#
#         if step % log_per_step == 0:
#             print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch + 1, epochs, i, len(train_loader),
#                                                                        total_loss))
#             total_loss = 0
#
#         del inputs, targets
#
#     # 一个epoch后，使用过验证集进行验证
#     accuracy, validation_loss = validate()
#     print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch + 1, accuracy, validation_loss))
#     torch.save(model, model_dir / f"model_{epoch}.pt")
#
#     # 保存最好的模型
#     if accuracy > best_accuracy:
#         torch.save(model, model_dir / f"model_best.pt")
#         best_accuracy = accuracy

# 加载最好的模型，然后进行测试集的预测
```

1. **开始训练：** 在未被注释的代码部分，我们可以看到关于模型训练的代码。在这部分代码中，使用了一个外层的`for`循环来遍历训练的多个epoch（训练周期）。每个epoch内部使用一个`for`循环遍历训练数据加载器`train_loader`中的每个小批量数据。
2. **模型训练：** 在内部的循环中，首先从数据加载器中获取一个批次的数据。然后，将这些数据传入模型进行前向传递得到预测输出，并计算预测输出与真实标签之间的损失（loss）。接着，调用反向传播（backpropagation）计算梯度，并利用优化器（`optimizer`）来更新模型的参数。
3. **打印loss：** 在内部循环的每一步（`step`）之后，代码会根据预先定义的`log_per_step`来打印当前的总损失（`total_loss`）。这样可以帮助我们监控训练的进展。
4. **验证：** 在每个epoch结束后，使用验证集数据来验证当前模型在验证集上的性能。调用`validate()`函数来实现验证。在`validate()`函数中，模型会切换到评估模式（`model.eval()`），然后对验证集的每个批次数据进行前向传递，并计算损失和正确预测的数量。最后，返回验证集上的准确率和平均损失。
5. **保存模型：** 在每个epoch结束后，将当前的模型保存到文件系统中，使用`torch.save()`函数。模型文件名包含epoch的编号，用于后续选择最佳模型进行测试。
6. **记录最佳模型：** 在每个epoch结束后，检查当前epoch的验证集准确率是否优于之前的最佳准确率（`best_accuracy`）。如果是，则将当前模型保存为最佳模型，并更新`best_accuracy`的值。

## 文本分类

```python
# 加载最好的模型，然后进行测试集的预测
model = torch.load(model_dir / f"model_best.pt")
model = model.eval()

test_dataset = MyDataset('test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
```

加载最佳模型，加载测试集数据并进行文本分类预测，完成任务一

## 提取关键词

```python
test_data['text'] = test_data['title'].fillna('') + ' ' + test_data['author'].fillna('') + ' ' + test_data[
    'abstract'].fillna('')

stops= stopwords.words('english')

model=SentenceTransformer(r'xlm-r-distilroberta-base-paraphrase-v1')

test_words=[]
for row in test_data.iterrows():
    # 读取第每一行数据的标题与摘要并提取关键词
    # 修改n_gram_range来改变结果候选词的词长大小。例如，如果我们将它设置为(3，3)，那么产生的候选词将是包含3个关键词的短语。
    n_gram_range = (2, 2)
    # 这里我们使用TF-IDF算法来获取候选关键词
    count = TfidfVectorizer(ngram_range=n_gram_range, stop_words=stops).fit([row[1].text])
    candidates = count.get_feature_names_out()
    # 将文本标题以及候选关键词/关键短语转换为数值型数据（numerical data）。我们使用BERT来实现这一目的
    title_embedding = model.encode([row[1].title])

    candidate_embeddings = model.encode(candidates)

    # 通过修改这个参数来更改关键词数量
    top_n = 15
    # 利用文章标题进一步提取关键词
    distances = cosine_similarity(title_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    if len(keywords) == 0:
        keywords = ['A', 'B']
    test_words.append('; '.join(keywords))

    print(f'success {row}')

test_data['Keywords'] = test_words
test_data[['uuid','Keywords','label']].to_csv('result.csv',index=False)
```

文本预处理和关键词抽取，并将提取的关键词结果保存到`result.csv`文件中。

让我们逐步解释代码的功能：

1. **文本预处理：** 首先，通过`test_data['title'].fillna('')`，`test_data['author'].fillna('')` 和 `test_data['abstract'].fillna('')` 分别对标题、作者和摘要列进行NaN值填充。这样可以确保所有文本数据都是字符串类型，并填充缺失的部分为空字符串。然后，通过拼接这三列的文本数据，并将结果存储在新的`text`列中，`test_data['text'] = ...`。

2. **停用词处理：** 使用NLTK库中的`stopwords.words('english')`函数获取英文停用词列表，存储在`stops`变量中。停用词是在信息检索中被忽略的常见词汇，这些词汇通常不携带太多实际意义，例如"the"、"a"、"and"等。

3. **关键词抽取：** 通过迭代`test_data`数据表中的每一行数据（样本），对每个样本的标题和摘要进行关键词抽取。其中，`test_data.iterrows()`是一个迭代器，用于遍历数据表的每一行。

4. **TF-IDF特征提取：** 对于每个样本，首先根据设置的`n_gram_range`，使用TF-IDF算法从文本中提取候选关键词。`TfidfVectorizer`是scikit-learn库中的一个工具，用于计算TF-IDF特征。

5. **BERT嵌入：** 使用预训练的`xlm-r-distilroberta-base-paraphrase-v1`模型，即`SentenceTransformer`，对样本的标题进行BERT嵌入编码。这样，每个样本的标题都被转换为一个向量表示。

6. **计算相似度：** 计算候选关键词的嵌入向量与标题嵌入向量之间的余弦相似度。相似度计算后，根据相似度的大小，选择相似度最高的前`top_n`个关键词作为最终的关键词。

7. **处理提取结果：** 如果从标题中没有提取到关键词（即`len(keywords) == 0`），则设置默认的关键词为 `['A', 'B']`。将提取到的关键词转换成一个用分号分隔的字符串，并将该字符串添加到`test_words`列表中。

8. **保存结果：** 将提取的关键词列表 `test_words` 添加到`test_data`数据表的`Keywords`列中。然后从`test_data`数据表中选取`uuid`、`Keywords`和`label`列，并将结果保存为`result.csv`文件，用于后续的提交和分析。

## 总结

`upper`完全体跑出来的分数相当可观，能够达到0.41792。

对于任务二，抽取式关键词应该已经到达极限了，再要提分就需要通过理解语义概括得到关键词，这也就转向了生成式关键词，也就是再进阶就需要用到LLM来冲榜了。

