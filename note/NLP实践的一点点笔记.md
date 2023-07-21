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

对于第二段代码生成的`submit_task2.csv`，结果存在一定问题，还需要进行相关debug