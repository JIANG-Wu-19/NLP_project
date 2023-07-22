# NLP实践的一点点笔记

友情连接：[Datawhale-NLP实践](https://datawhaler.feishu.cn/docx/WirRd4oB5oMe2ixw1rxcTnHFnHh)

## 直播一

### 环境配置

* miniconda的安装，本人选择了anaconda的安装，本质上来说miniconda是anaconda的轻量版，anaconda完整性更好
* 对于镜像源，本人并没有进行替换，增加了清华源

### pytorch环境的搭建



* 创建conda环境

  ```
  conda create -n pytorch python=3.10
  ```

  环境名称：pytorch

  python版本：3.10

* 激活pytorch环境

  ```
  conda activate pytorch
  ```

  从base->pytorch

* 安装pytorch

  先在**Anaconda Powershell Prompt**中输入：

  ```
  nvidia-smi
  ```

  得到如下图片：

  ![image-20230721163031733](img\1.png)

  CUDA Version=12.0，因此我的pytorch-cuda版本最高不能超过12.0，在这里安装了11.8版本：

  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

* 验证pytorch是否安装成功

  进入pytorch环境，键入python进入python编译器，输入：

  ```python
  import torch
  torch.cuda.is_available()
  ```

  返回`True`则安装成功

## 直播二

### 任务

[基于论文摘要的文本分类与关键词抽取挑战赛](https://challenge.xfyun.cn/topic/info?type=abstract-of-the-paper&ch=ZuoaKcY)

* task 1

  判断是否是医学论文

* task 2

  关键词抽取

### baseline

1. 传统方法
2. BERT预训练模型
3. 微调chatGLM2-6B

## 跑通baseline

首先fork了其中一个[baseline](https://aistudio.baidu.com/aistudio/projectdetail/6522950?sUid=377372&shared=1&ts=1689827255213)，运行在百度AI Studio上，选择`V100 16G`运行

运行完之后，提交`submit_task1.csv`，获得分数0.99384

~~对于第二段代码生成的`submit_task2.csv`，结果存在一定问题，还需要进行相关debug~~

## 对baseline进行fine-tune

将baseline下载到本地，对于Logistic回归模型进行替换，替换模型有：SVM（支持向量机）、随机森林模型，对baseline进行相关代码替换

结果显示，在default参数的情况下，Logistic回归模型表现更好，Random Forest次之，SVM表现最差，因此这条选择不同分类器的优化路径行不通。

仍然基于baseline，对LogisticRegression模型的超参数进行调参优化，使用交叉验证技术来找到最佳的参数组合，将得到的参数组合重新应用于baseline，得到的结果略微优于原结果



> To be continued

## 更新日志

2023年7月21日：跑通了baseline

2023年7月22日：对baseline进行了fine-tune，实际结果略微优于baseline；尝试使用BERT模型，但是从运行中输出的部分参数来看，结果不尽如人意，同时显存爆了，建议上云