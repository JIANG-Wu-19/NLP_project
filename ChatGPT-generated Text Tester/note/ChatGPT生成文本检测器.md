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

 后续尝试方向：

 * 调用BERT模型进行鉴别
 * 调用大模型进行鉴别

 ## 使用BERT模型进行分类

 简单了解一下BERT模型

 ### BERT模型

 BERT（Bidirectional Encoder Representations from Transformers） 是一种预训练的自然语言处理模型，由Google研发并于2018年发布。BERT采用了**Transformer网络结构**，通过大规模无监督训练从而学习到单词或字符级别的语言表示。相比于传统的基于循环神经网络的模型，BERT采用了双向编码器（Bidirectional Encoder）的思想，可以同时利用上下文中的信息来理解单词的语义和含义。

 BERT的预训练过程分为两个阶段：**Masked Language Model（MLM）**和**Next Sentence Prediction（NSP）**。在MLM阶段，BERT会随机遮盖输入文本中的部分单词，并尝试预测这些遮盖的单词是什么。在NSP阶段，BERT会输入两个句子，并预测这两个句子是否是连续的。

 预训练后，BERT可以通过微调（Fine-tuning）的方式在特定任务上进行训练，例如文本分类、命名实体识别等。通过微调，BERT可以根据不同任务的数据进行适应性学习，提取有关任务的上下文相关特征，从而更好地完成特定的自然语言处理任务。

 BERT的出现填补了自然语言处理领域在预训练模型上的空白，它在多个基准数据集上取得了显著的性能提升，并且对于多种自然语言处理任务都具有泛化能力。BERT的成功也促进了许多后续的预训练模型的发展，为自然语言处理的研究和应用提供了重要的推动力。

 BERT模型的结构主要由Transformer网络组成，它由多个编码器层叠加而成。每个编码器层由两个子层组成，分别是多头自注意力机制（Multi-Head Self-Attention）和前馈神经网络（Feed-Forward Neural Network）。下面我们逐步解释BERT模型的结构和各个组件的作用。

 1. 输入表示：BERT的输入表示采用了词嵌入和位置编码的结合。首先，对输入的文本进行分词处理，然后将每个分词映射为一个固定维度的词向量。位置编码会为每个词向量附加一个位置信息，以捕捉单词在句子中的相对位置关系。
 2. 编码器层：BERT模型通常包含多个相同结构的编码器层，每个编码器层都由两个子层组成。
    * 多头自注意力机制（Multi-Head Self-Attention）：在这个子层中，输入的序列经过一系列的自注意力计算，从而学习到每个单词与其他单词之间的相互作用和关联。通过引入多个注意力头，可以并行地学习多个表示，增加模型的泛化能力。
    * 前馈神经网络（Feed-Forward Neural Network）：在这个子层中，每个位置的隐藏表示会经过两层全连接网络进行非线性变换，增强模型的表示能力。前馈神经网络采用了激活函数（如ReLU）来引入非线性。

 3. 预训练目标：
    * 遮盖语言模型（Masked Language Model，MLM）：BERT采用遮盖部分输入单词的方式，然后通过预测被遮盖单词的方式进行训练。这使得模型能够学习到单词之间的上下文信息，从而更好地理解句子。
    * 下一句预测（Next Sentence Prediction，NSP）：BERT的预训练还包括判断两个句子之间是否连续的任务。通过这个任务，模型可以学习到句子之间的语义关系和连接方式。

 BERT模型的核心思想是通过大规模的无监督预训练，学习到通用的句子表示，然后通过微调的方式在具体任务上进行训练。预训练过程使得BERT模型具备了更好的语义理解能力和句子关系建模能力，从而在各种自然语言处理任务中取得了显著的性能提升。

 ### BERT的思路

 关于使用BERT模型，之前的[项目]([NLP_project/Text classification and keyword extraction based on abstracts at master · JIANG-Wu-19/NLP_project (github.com)](https://github.com/JIANG-Wu-19/NLP_project/tree/master/Text classification and keyword extraction based on abstracts))已经实现过，基于BERT模型判断文本是否属于医学论文，也就是说可以依靠原有代码，在原有代码的基础上进行相关改动就可以实现了，关于BERT模型进行文本分类脚本的实现，可以在《[upper的实现 | J&Ocean BLOG](https://jiang-wu-19.github.io/2023/07/30/upper的实现/)》这篇blog中进一步了解

 BERT模型的训练可以在本地实现，考虑到笔者的配置有限，将batch_size设置为10，运行时只需要使用3.8G的显存。

 BERT跑出来的结果还不错，高于原有的进阶分数，达到0.93+

 ## 不死心，仍要使用大模型的API

 事实证明，API最多只能回答一些生成式的问题，也许是API背后的模型是综合类型的，并没有经过二分类的fine-tune，代码确实写好了，但是由于鉴别的文本相当冗长，每进行一次request耗费大量的tokens还有时间

 ~~对API的调用感兴趣的可以看一看失败的的代码~~

 ## 选择ernie模型，取得阶段性小胜利

 使用paddle框架和ernie模型

 ### paddlepaddle和

 PaddlePaddle（百度飞桨）是百度开发的开源深度学习框架，旨在为科研人员和工程师提供高效、灵活、全面的深度学习平台。它在深度学习领域具有广泛的应用，特别在自然语言处理、计算机视觉、语音识别等领域表现出色。以下是PaddlePaddle框架的一些关键特点和组成部分的简要介绍：

 1. **动态图和静态图支持**：PaddlePaddle支持动态图和静态图两种计算模式。动态图适用于快速原型设计和交互式调试，而静态图则可以优化计算图以提高性能和效率。

 2. **高性能优化**：PaddlePaddle针对多种硬件和平台进行了高性能优化，包括CPU、GPU、FPGA等。它采用了诸如自动混合精度、异步数据加载等技术，以提高训练和推理的速度。

 3. **多样的模型库和工具**：PaddlePaddle提供了丰富的预训练模型和模型库，涵盖自然语言处理、计算机视觉、推荐系统等多个领域。同时，它还提供了数据处理、模型评估、可视化等一系列工具，方便用户进行全面的深度学习任务。

 4. **易用性和灵活性**：PaddlePaddle设计了简单易用的API，使得初学者可以快速上手。同时，它也提供了灵活的自定义功能，以满足高级用户的需求。

 5. **分布式训练和部署**：PaddlePaddle支持分布式训练，可以在多个机器和设备上并行进行训练。此外，它还提供了模型转换和部署工具，使模型部署更加便捷。

 6. **自然语言处理工具包**：PaddlePaddle自然语言处理工具包（PaddleNLP）为处理文本数据提供了丰富的功能，包括预训练模型、分词、命名实体识别、文本分类等。

 7. **社区支持和文档**：PaddlePaddle有活跃的社区，提供了丰富的教程、文档和示例代码，帮助用户更好地了解和使用框架。

 ### paddleNLP

 PaddleNLP是百度飞桨（PaddlePaddle）深度学习框架的自然语言处理（NLP）工具包，专门为处理文本数据和解决自然语言处理问题而设计。它提供了丰富的预训练模型、数据处理工具和任务特定的API，使得NLP任务的开发和研究更加便捷和高效。以下是PaddleNLP的一些主要特点和组成部分的简要介绍：

 1. **预训练模型库**：PaddleNLP提供了丰富的预训练模型，包括BERT、ERNIE、RoBERTa、GPT等。这些模型在大规模文本数据上进行预训练，可以用于多种NLP任务的迁移学习和微调。

 2. **任务特定API**：PaddleNLP为常见的NLP任务（如文本分类、序列标注、句子对匹配等）提供了专门的API接口，使得模型的训练和评估变得更加简单。开发者可以直接使用这些API来完成各种任务，无需从头开始构建模型。

 3. **分词和数据处理工具**：PaddleNLP提供了强大的分词和数据处理工具，用于将原始文本数据转换为模型可以处理的格式。这些工具支持中文、英文等多种语言，帮助用户有效地准备数据集。

 4. **模型可解释性工具**：PaddleNLP还提供了模型可解释性工具，可以帮助用户理解模型的预测结果，分析哪些部分影响了模型的决策。

 5. **多语言支持**：PaddleNLP支持多种语言，包括中文和英文，使其适用于全球范围的NLP研究和应用。

 6. **与PaddlePaddle集成**：PaddleNLP与PaddlePaddle深度学习框架无缝集成，用户可以直接将PaddleNLP的模型和功能与PaddlePaddle的其他组件结合使用。

 ### Ernie模型

 ERNIE（Enhanced Representation through kNowledge IntEgration）是一种基于Transformer架构的预训练语言模型，由百度研究团队开发。ERNIE的网络架构与Transformer的基本结构类似，但在细节上进行了一些创新和改进，以适应多领域、多语种的知识融合。以下是ERNIE模型的网络架构的简要介绍：

 1. **Transformer基本结构**：ERNIE模型采用了Transformer架构，它由多个编码器层和解码器层组成。每个编码器和解码器层包含自注意力机制和前馈神经网络。自注意力机制有助于模型捕捉输入序列中不同位置之间的依赖关系。

 2. **多头注意力**：类似于Transformer，ERNIE的注意力机制也采用了多头注意力。这使得模型可以在不同的注意力头上并行学习不同的信息，从而更好地捕捉序列中的各种关系。

 3. **连续字词表示**：ERNIE引入了连续字词表示，对于中文等没有明确分词边界的语言尤为重要。它通过将字母、音节等形式进行建模，以更好地捕获复杂的语言结构。

 4. **双向语言模型预训练**：ERNIE使用了双向语言模型（Bidirectional Language Model，BiLM）进行预训练。这意味着模型在预测当前词语时，可以利用前面和后面的词语信息，从而获得更好的上下文表示。

 5. **多领域、多语种知识融合**：ERNIE的创新之一是将多领域、多语种的知识融合到预训练中。它通过同时在多个领域和多种语言的数据上进行预训练，使模型能够学习丰富的知识和表示。

 6. **多任务学习**：ERNIE在预训练和微调阶段使用了多任务学习。这意味着模型在同一时间可以同时学习多个任务，从而增强模型的泛化能力和效果。

 ### 解决思路

 使用PaddlePaddle深度学习框架和PaddleNLP自然语言处理工具包构建文本分类模型。该模型用于将文本分类为两个类别，并通过训练数据进行模型训练和验证，然后使用该模型对测试数据进行分类，并生成一个提交文件。

 以下是代码的详细分析：

 1. **导入库**：首先，导入了必要的库，包括NumPy（用于数值计算）、Pandas（用于数据分析）、tqdm（用于显示进度条）、PaddlePaddle（深度学习框架）、PaddleNLP（用于自然语言处理）、以及其他用于数据加载、模型训练等的库。

 2. **加载数据**：使用Pandas库从CSV文件中加载训练数据和测试数据。

 3. **载入模型与分词器**：使用`AutoModelForSequenceClassification`从预训练的ernie-3.0-mini-zh模型中加载序列分类模型，并设置分类类别数为2。同样，使用`AutoTokenizer`加载相应的分词器。

 4. **定义优化器和损失函数**：定义AdamW优化器和交叉熵损失函数。

 5. **划分训练集和验证集**：使用`train_test_split`将训练数据划分为训练集和验证集。

 6. **数据处理与加载**：将数据转换为适合数据加载器的格式，并创建数据加载器。训练数据可以随机打乱，以便模型更好地学习。注意，这里的数据加载器用的是PaddlePaddle的`DataLoader`，可以在训练过程中并行加载数据，提高训练效率。

 7. **模型训练循环**：使用循环进行模型训练，训练多个epochs。在每个epoch内，模型会被切换到训练模式，然后遍历训练数据批次进行训练。具体步骤包括：

    - 将文本数据转换为模型可以处理的格式，包括使用分词器进行分词并填充到固定长度。
    - 将数据输入模型，得到预测结果。
    - 计算预测结果与真实标签之间的交叉熵损失。
    - 执行反向传播以计算梯度。
    - 使用优化器更新模型参数。

 8. **验证过程**：在每个epoch结束后，切换模型为验证模式，并使用验证数据集计算验证损失。同样，将数据转换为模型可以处理的格式，进行前向传播计算，并计算验证损失。

 9. **保存最佳模型**：如果当前epoch的准确率优于之前的最佳准确率，将模型和优化器参数保存为检查点文件。

 10. **模型保存**：在所有epoch训练完成后，将最终模型和优化器参数保存。

 11. **模型推理**：定义了一个用于对输入文本进行预测的函数`infer`，该函数将输入文本转换为模型可以处理的格式，然后使用训练好的模型进行预测，并返回预测结果。

 12. **对测试数据进行预测**：使用上述`infer`函数将测试数据的文本送入模型，得到预测结果。然后创建一个包含预测结果的新数据表，准备生成提交文件。

 13. **生成提交文件**：将包含预测结果的测试数据表保存为CSV文件，用于提交到竞赛平台。

 

 在使用Ernie模型的时候，为了加快速度，放弃了本地环境，使用了百度AI Studio云环境进行训练，每一轮训练时间为60s，验证时间为6s

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

 ### bert.py

 ```python
 # 导入前置依赖
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
 dataset_dir = Path("./ChatGPT生成文本检测器")
 os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else ''
 
 # 模型存储路径
 model_dir = Path("./model/bert_checkpoints")
 # 如果模型目录不存在，则创建一个
 os.makedirs(model_dir) if not os.path.exists(model_dir) else ''
 
 print("Device:", device)
 
 # 读取数据集，进行数据处理
 
 pd_train_data = pd.read_csv('train.csv')
 pd_train_data['content']=pd_train_data['content'].fillna('')
 
 test_data = pd.read_csv('test.csv')
 test_data['content']=test_data['content'].fillna('')
 
 
 # 从训练集中随机采样测试集
 validation_data = pd_train_data.sample(frac=validation_ratio)
 train_data = pd_train_data[~pd_train_data.index.isin(validation_data.index)]
 
 
 # 构建Dataset
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
         text = data['content']
         # 根据状态返回内容
         if self.mode == 'test':
             # 如果是test，将uuid做为target
             label = data['name']
         else:
             label = data['label']
         # 返回内容和label
         return text, label
 
     def __len__(self):
         return len(self.dataset)
 
 
 train_dataset = MyDataset('train')
 validation_dataset = MyDataset('validation')
 
 train_dataset.__getitem__(0)
 
 # 获取Bert预训练模型
 tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
 
 
 # 接着构造我们的Dataloader。
 # 我们需要定义一下collate_fn，在其中完成对句子进行编码、填充、组装batch等动作：
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
 
 
 train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
 validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
 
 inputs, targets = next(iter(train_loader))
 print("inputs:", inputs)
 print("targets:", targets)
 
 
 # 定义预测模型，该模型由bert模型加上最后的预测层组成
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
 
 
 model = MyModel()
 model = model.to(device)
 
 # 定义出损失函数和优化器。这里使用Binary Cross Entropy：
 criteria = nn.BCELoss()
 optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
 
 # 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
 def to_device(dict_tensors):
     result_tensors = {}
     for key, value in dict_tensors.items():
         result_tensors[key] = value.to(device)
     return result_tensors
 
 
 # 定义一个验证方法，获取到验证集的精准率和loss
 def validate():
     model.eval()
     total_loss = 0.
     total_correct = 0
     for inputs, targets in validation_loader:
         inputs, targets = to_device(inputs), targets.to(device)
         outputs = model(inputs)
         loss = criteria(outputs.view(-1), targets.float())
         total_loss += float(loss)
 
         correct_num = (((outputs = 0.5).float() * 1).flatten() == targets).sum()
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
 for epoch in range(epochs):
     model.train()
     for i, (inputs, targets) in enumerate(train_loader):
         # 从batch中拿到训练数据
         inputs, targets = to_device(inputs), targets.to(device)
         # 传入模型进行前向传递
         outputs = model(inputs)
         # 计算损失
         loss = criteria(outputs.view(-1), targets.float())
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()
 
         total_loss += float(loss)
         step += 1
 
         if step % log_per_step == 0:
             print("Epoch {}/{}, Step: {}/{}, total loss:{:.4f}".format(epoch + 1, epochs, i, len(train_loader),
                                                                        total_loss))
             total_loss = 0
 
         del inputs, targets
 
     # 一个epoch后，使用过验证集进行验证
     accuracy, validation_loss = validate()
     print("Epoch {}, accuracy: {:.4f}, validation loss: {:.4f}".format(epoch + 1, accuracy, validation_loss))
     torch.save(model, model_dir / f"model_{epoch}.pt")
 
     # 保存最好的模型
     if accuracy  best_accuracy:
         torch.save(model, model_dir / f"model_best.pt")
         best_accuracy = accuracy
 
 # 加载最好的模型，然后进行测试集的预测
 model = torch.load(model_dir / f"model_best.pt")
 model = model.eval()
 
 test_dataset = MyDataset('test')
 test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
 
 results = []
 for inputs, ids in test_loader:
     outputs = model(inputs.to(device))
     outputs = (outputs = 0.5).int().flatten().tolist()
     ids = ids.tolist()
     results = results + [(id, result) for result, id in zip(outputs, ids)]
 
 test_label = [pair[1] for pair in results]
 test_data['label'] = test_label
 
 test_data[['name','label']].to_csv('result.csv',index=False)
 
 ```

 ### api.py

 ```python
 import pandas as pd
 import copy
 import zhipuai
 import time
 from tqdm import tqdm
 
 train = pd.read_csv('train.csv')
 test = pd.read_csv('test.csv')
 
 train['content'] = train['content'].apply(lambda x: x[1:-1])
 test['content'] = test['content'].apply(lambda x: x[1:-1])
 
 zhipuai.api_key = "YOUR_API_KEY"
 
 n = 5
 prompt_data = train.sample(n=n)
 prompt_data = prompt_data.reset_index(drop=True)
 
 print(prompt_data)
 
 tmp = []
 for i in range(len(prompt_data)):
     item = prompt_data.loc[i]
     instruction = {
         "role": "user",
         "content": f"将给定的文本分成两类，并给出标签0或1，给定文本如下：{item[2]}"
     }
     answer = {
         "role": "assistant",
         "content": f"{item[1]}"
     }
     tmp.append(instruction)
     tmp.append(answer)
 
 print(tmp)
 
 test_item = test.loc[0]
 
 
 def predict(test_item, tmp):
     prompt = copy.deepcopy(tmp)
     test_instruction = {
         "role": "user",
         "content": f"将给定的文本分成两类，并给出标签0或1，给定文本如下：{test_item[1]}"
     }
     prompt.append(test_instruction)
     print(prompt)
     # response = zhipuai.model_api.invoke(
     #     model="chatglm_pro",
     #     prompt=prompt
     # )
     # print(response)
 
 
 predict(test_item=test_item, tmp=tmp)
 
 ```

 ### ernie.py

 ```python
 import numpy as np # 数值计算
 import pandas as pd # 数据分析
 from tqdm import tqdm # 进度条显示
 import paddle # PaddlePaddle 深度学习框架
 from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer # 飞桨自然语言处理工具包（模型、分词器）
 from paddle.io import DataLoader # 数据加载器
 from paddlenlp.datasets import MapDataset # 数据集转换
 from sklearn.model_selection import train_test_split # 训练集与验证集拆分
 import matplotlib.pyplot as plt # 绘图
 
 data = pd.read_csv("/home/aistudio/data/ChatGPT生成文本检测器公开数据-更新/train.csv") # 加载赛事提供的训练数据
 test_data = pd.read_csv("/home/aistudio/data/ChatGPT生成文本检测器公开数据-更新/test.csv") # 加载赛事所需提交的测试数据
 data.sample(frac=1).head() # 随机查看 5 行训练数据中的内容
 
 # 载入模型与分词器
 
 # 使用 ernie-3.0-mini-zh 序列分类模型，并将分类类别数设置为 2
 model = AutoModelForSequenceClassification.from_pretrained("ernie-3.0-mini-zh", num_classes=2)
 # 使用 ernie-3.0-mini-zh 分词器
 tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-mini-zh")
 
 # 定义 AdamW 优化器，学习率为 0.000001
 optimizer = paddle.optimizer.AdamW(1e-5, parameters=model.parameters())
 
 # 定义损失函数为交叉熵函数，计算每个 mini batch 的均值
 loss_fn = paddle.nn.loss.CrossEntropyLoss(reduction='mean')
 
 # 按照 10% 的比例划分训练集与验证集
 train_data, valid_data = train_test_split(data, test_size=0.1)
 
 # 下面就是一堆操作，把数据变成数据加载器可以识别的格式，自定义数据集类也是同样的效果
 train_dict = train_data.to_dict(orient='records')
 valid_dict = valid_data.to_dict(orient='records')
 train_ds = MapDataset(train_dict)
 valid_ds = MapDataset(valid_dict)
 
 # 将整体数据拆分为 30 份，放入数据加载器，就是一次性会有 <总样本数 / 30 份数据同时并行计算，份数越多，并行越多，显存占用越大，需要根据需求来选择
 train_loader = DataLoader(train_dict, batch_size=60, shuffle=True) # 训练数据可以随机打乱，让模型更好地学习，减轻学习到无关特征的问题
 valid_loader = DataLoader(valid_dict, batch_size=60) # 这里用的是 V100 32G，如果是显存更小的卡，需要调小一点，不然会炸显存
 
 epochs=200
 best_accuracy=0
 for epoch in range(epochs): # 训练 30 轮
     # 训练过程
     model.train() # 切换模型为训练模式
     for batch_x in tqdm(train_loader): # 每次从数据加载器读入一批(batch)数据
         X = tokenizer(batch_x["content"], max_length=1015, padding=True) # 将数据转换为模块可处理的数据形式
         input_ids = paddle.to_tensor(X['input_ids'], dtype="int32") # 将 input_ids 变为张量，方便并行计算
         token_type_ids = paddle.to_tensor(X['token_type_ids'], dtype="int32") # 将 token_type_ids 变为张量
         pred = model(input_ids, token_type_ids) # 将数据读入模型，并得到计算后的结果
         loss = loss_fn(pred, paddle.to_tensor(batch_x["label"], dtype="int32")) # 对比预测结果与真实结果，计算损失函数的值
         loss.backward() # 反向传播，计算梯度
         optimizer.step() # 优化器根据梯度与学习率调整模型参数
         optimizer.clear_gradients() # 清空梯度，避免下次梯度计算时累加
 
     # 验证过程
     model.eval() # 切换模型为验证模式
     val_loss = [] # 验证集数据的损失函数合集
     with paddle.no_grad(): # 在模型验证时，只做前向计算，因此不需要保存梯度信息
         for batch_x in tqdm(valid_loader): # 下面的操作与训练过程相同
             X = tokenizer(batch_x["content"], max_length=1015, padding=True)
             input_ids = paddle.to_tensor(X['input_ids'], dtype="int32")
             token_type_ids = paddle.to_tensor(X['token_type_ids'], dtype="int32")
             pred = model(input_ids, token_type_ids)
             loss = loss_fn(pred, paddle.to_tensor(batch_x["label"], dtype="int32"))
             val_loss.append(loss.item()) # 将计算出的损失函数值存入合集
             
     # 打印本轮训练的验证集损失函数值，与预测正确率
     accuracy = (pred.argmax(1) == batch_x["label"]).astype('float').mean().item()
     print('Epoch {0}, Val loss {1:3f}, Val Accuracy {2:3f}'.format(
     epoch,
     np.mean(val_loss), 
     (pred.argmax(1) == batch_x["label"]).astype('float').mean().item()
 ))
     if accuracybest_accuracy:
         paddle.save(model.state_dict(),"/home/aistudio/work/model_best.pdparams")
         paddle.save(optimizer.state_dict(), "/home/aistudio/work/optimizer_best.pdopt")
         best_accuracy=accuracy
 
 paddle.save(model.state_dict(),"/home/aistudio/work/model.pdparams")
 paddle.save(optimizer.state_dict(), "/home/aistudio/work/optimizer.pdopt")
 
 # 如果你拿到了模型参数（在 AIStudio 中提供），你可以运行这行代码，如果直接运行模型，则没有必要运行
 
 # 载入模型参数、优化器参数的最后一个epoch保存的检查点
 layer_state_dict = paddle.load("/home/aistudio/work/model_best.pdparams")
 opt_state_dict = paddle.load("/home/aistudio/work/optimizer_best.pdopt")
 
 # 将加载后的参数与模型关联起来
 model.set_state_dict(layer_state_dict)
 optimizer.set_state_dict(opt_state_dict)
 
 def infer(string: str) - int:
     """将文本传入模型并返回预测结果
     
     输入：
         - string: str
             待预测的文本内容
     
     输出:
         - result: int
             模型的预测结果
     """
     X = tokenizer([string], max_length=1015, padding=True)
     input_ids = paddle.to_tensor(X['input_ids'], dtype="int32")
     token_type_ids = paddle.to_tensor(X['token_type_ids'], dtype="int32")
     pred = model(input_ids, token_type_ids)
     result = pred.argmax(1).item() # 获取预测概率最大的那个类别
     return result
 
 test_data["label"] = test_data["content"].apply(infer) # 将测试集的每个文本送入模型返回结果
 submit = test_data.drop(columns=["content"]) # 生成提交数据（就是把带结果的测试集丢掉内容，复制一份）
 submit.to_csv("submit3.csv", index=False) # 保存 CSV 文件
 ```

 

 ## 更新日志

 2023年8月4日：跑通baseline并进行微调，收效甚微甚至不如baseline，仅提交了baseline，获得分数0.85+

 2023年8月5日：跑通upper，获得分数0.91+

 2023年8月6日：尝试使用BERT模型进行鉴别，效果比较好，获得分数0.93+

 2023年8月7日：尝试使用大模型API进行分类，效果相当不理想；尝试使用ernie模型进行分类，分类效果随着训练轮数的增加而有一定程度的提升，得到0.98+、0.99+分数

 

 

  To Be Continued