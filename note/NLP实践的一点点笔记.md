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

  ![image-20230721163031733](img/1.png)

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

## 使用BERT模型进行分类

使用了BERT模型进行预训练

相关参数进行调整：

* 考虑到本地的硬件配置（4GB显存），采用batch_xize=16，在第二轮训练中就会爆显存；设置batch_size=10
* 为了防止数据过拟合，设置epoch=10

训练过程如下：

```
Device: cuda
inputs: {'input_ids': tensor([[  101,  2470,  2006,  ...,  6742,  4118,   102],
        [  101,  2070,  2047,  ...,  2951,  4106,   102],
        [  101, 12509,  1011,  ...,  3728,  2764,   102],
        ...,
        [  101, 11968,  6767,  ...,  1998,  3707,   102],
        [  101,  3935,  8360,  ...,  2256,  9896,   102],
        [  101, 11742,  6047,  ...,  1006, 24529,   102]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1]])}
targets: tensor([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
Epoch 1/10, Step: 49/540, total loss:15.7107
Epoch 1/10, Step: 99/540, total loss:7.6416
Epoch 1/10, Step: 149/540, total loss:6.4683
Epoch 1/10, Step: 199/540, total loss:7.7938
Epoch 1/10, Step: 249/540, total loss:6.7497
Epoch 1/10, Step: 299/540, total loss:6.3569
Epoch 1/10, Step: 349/540, total loss:5.7305
Epoch 1/10, Step: 399/540, total loss:7.5373
Epoch 1/10, Step: 449/540, total loss:3.2843
Epoch 1/10, Step: 499/540, total loss:5.0083
Epoch 1, accuracy: 0.9567, validation loss: 0.0112
Epoch 2/10, Step: 9/540, total loss:6.4998
Epoch 2/10, Step: 59/540, total loss:3.8504
Epoch 2/10, Step: 109/540, total loss:4.3037
Epoch 2/10, Step: 159/540, total loss:3.7314
Epoch 2/10, Step: 209/540, total loss:2.8143
Epoch 2/10, Step: 259/540, total loss:2.6459
Epoch 2/10, Step: 309/540, total loss:4.2030
Epoch 2/10, Step: 359/540, total loss:4.3778
Epoch 2/10, Step: 409/540, total loss:3.6712
Epoch 2/10, Step: 459/540, total loss:3.7527
Epoch 2/10, Step: 509/540, total loss:4.8828
Epoch 2, accuracy: 0.9717, validation loss: 0.0076
Epoch 3/10, Step: 19/540, total loss:4.3283
Epoch 3/10, Step: 69/540, total loss:1.9251
Epoch 3/10, Step: 119/540, total loss:3.9258
Epoch 3/10, Step: 169/540, total loss:3.9836
Epoch 3/10, Step: 219/540, total loss:2.5203
Epoch 3/10, Step: 269/540, total loss:3.0695
Epoch 3/10, Step: 319/540, total loss:3.0718
Epoch 3/10, Step: 369/540, total loss:2.8674
Epoch 3/10, Step: 419/540, total loss:1.6683
Epoch 3/10, Step: 469/540, total loss:3.6798
Epoch 3/10, Step: 519/540, total loss:2.5205
Epoch 3, accuracy: 0.9667, validation loss: 0.0095
Epoch 4/10, Step: 29/540, total loss:1.2602
Epoch 4/10, Step: 79/540, total loss:1.7367
Epoch 4/10, Step: 129/540, total loss:1.7730
Epoch 4/10, Step: 179/540, total loss:2.7801
Epoch 4/10, Step: 229/540, total loss:2.7113
Epoch 4/10, Step: 279/540, total loss:2.2387
Epoch 4/10, Step: 329/540, total loss:1.8035
Epoch 4/10, Step: 379/540, total loss:3.1783
Epoch 4/10, Step: 429/540, total loss:2.4875
Epoch 4/10, Step: 479/540, total loss:1.0889
Epoch 4/10, Step: 529/540, total loss:1.8472
Epoch 4, accuracy: 0.9717, validation loss: 0.0093
Epoch 5/10, Step: 39/540, total loss:1.4521
Epoch 5/10, Step: 89/540, total loss:2.7088
Epoch 5/10, Step: 139/540, total loss:1.3743
Epoch 5/10, Step: 189/540, total loss:0.9969
Epoch 5/10, Step: 239/540, total loss:3.5632
Epoch 5/10, Step: 289/540, total loss:1.6101
Epoch 5/10, Step: 339/540, total loss:0.6479
Epoch 5/10, Step: 389/540, total loss:1.6050
Epoch 5/10, Step: 439/540, total loss:1.4850
Epoch 5/10, Step: 489/540, total loss:1.8811
Epoch 5/10, Step: 539/540, total loss:1.7147
Epoch 5, accuracy: 0.9800, validation loss: 0.0079
Epoch 6/10, Step: 49/540, total loss:0.9615
Epoch 6/10, Step: 99/540, total loss:0.7971
Epoch 6/10, Step: 149/540, total loss:1.5873
Epoch 6/10, Step: 199/540, total loss:4.3790
Epoch 6/10, Step: 249/540, total loss:1.6141
Epoch 6/10, Step: 299/540, total loss:2.8523
Epoch 6/10, Step: 349/540, total loss:2.9342
Epoch 6/10, Step: 399/540, total loss:1.8583
Epoch 6/10, Step: 449/540, total loss:1.2382
Epoch 6/10, Step: 499/540, total loss:2.0442
Epoch 6, accuracy: 0.9633, validation loss: 0.0106
Epoch 7/10, Step: 9/540, total loss:1.9456
Epoch 7/10, Step: 59/540, total loss:0.3481
Epoch 7/10, Step: 109/540, total loss:0.7371
Epoch 7/10, Step: 159/540, total loss:1.0779
Epoch 7/10, Step: 209/540, total loss:0.5513
Epoch 7/10, Step: 259/540, total loss:0.6267
Epoch 7/10, Step: 309/540, total loss:0.6717
Epoch 7/10, Step: 359/540, total loss:0.0804
Epoch 7/10, Step: 409/540, total loss:0.7880
Epoch 7/10, Step: 459/540, total loss:1.2405
Epoch 7/10, Step: 509/540, total loss:0.3852
Epoch 7, accuracy: 0.9717, validation loss: 0.0113
Epoch 8/10, Step: 19/540, total loss:0.0597
Epoch 8/10, Step: 69/540, total loss:0.1654
Epoch 8/10, Step: 119/540, total loss:0.0882
Epoch 8/10, Step: 169/540, total loss:0.0558
Epoch 8/10, Step: 219/540, total loss:2.7450
Epoch 8/10, Step: 269/540, total loss:2.9093
Epoch 8/10, Step: 319/540, total loss:0.2964
Epoch 8/10, Step: 369/540, total loss:2.9049
Epoch 8/10, Step: 419/540, total loss:0.8974
Epoch 8/10, Step: 469/540, total loss:0.8773
Epoch 8/10, Step: 519/540, total loss:0.8961
Epoch 8, accuracy: 0.9783, validation loss: 0.0136
Epoch 9/10, Step: 29/540, total loss:0.6021
Epoch 9/10, Step: 79/540, total loss:0.2185
Epoch 9/10, Step: 129/540, total loss:0.0978
Epoch 9/10, Step: 179/540, total loss:0.8829
Epoch 9/10, Step: 229/540, total loss:0.9440
Epoch 9/10, Step: 279/540, total loss:0.4725
Epoch 9/10, Step: 329/540, total loss:0.8252
Epoch 9/10, Step: 379/540, total loss:0.2206
Epoch 9/10, Step: 429/540, total loss:0.2055
Epoch 9/10, Step: 479/540, total loss:1.2238
Epoch 9/10, Step: 529/540, total loss:0.3699
Epoch 9, accuracy: 0.9767, validation loss: 0.0137
Epoch 10/10, Step: 39/540, total loss:0.1256
Epoch 10/10, Step: 89/540, total loss:0.0799
Epoch 10/10, Step: 139/540, total loss:0.0696
Epoch 10/10, Step: 189/540, total loss:0.0255
Epoch 10/10, Step: 239/540, total loss:0.2567
Epoch 10/10, Step: 289/540, total loss:0.0501
Epoch 10/10, Step: 339/540, total loss:1.4652
Epoch 10/10, Step: 389/540, total loss:2.1342
Epoch 10/10, Step: 439/540, total loss:0.2545
Epoch 10/10, Step: 489/540, total loss:0.0748
Epoch 10/10, Step: 539/540, total loss:0.2158
Epoch 10, accuracy: 0.9633, validation loss: 0.0198
```

不难看出，第五轮训练的模型最好，跑了测试集之后，得到的分数是0.99425，有所提升。

## 更新了B集

7月24日0时，官方更新了数据集B，与原始测试集不同的是取消了Keywords那一列，因此需要对baseline进行调整。

然而，仅仅使用baseline的适应版本会出大问题，先不说Keywords是否能够提取到，最要命的是连文本分类都错误频出，分数仅有0.27164。

所以选择转换思路，选用BERT模型重新进行训练，本次训练过程如下：

```
Device: cuda
inputs: {'input_ids': tensor([[  101, 24655,  4997,  ...,  2057, 16599,   102],
        [  101,  2943,  4487,  ...,  4725,  2000,   102],
        [  101,  5090,  1998,  ...,  2050,  1010,   102],
        ...,
        [  101,  1996,  2110,  ...,  2192,  1010,   102],
        [  101, 11643,  2986,  ..., 11650,  4819,   102],
        [  101,  4254,  1997,  ...,  2426,  3633,   102]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1]])}
targets: tensor([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
Epoch 1/10, Step: 49/540, total loss:17.4060
Epoch 1/10, Step: 99/540, total loss:8.2182
Epoch 1/10, Step: 149/540, total loss:4.8785
Epoch 1/10, Step: 199/540, total loss:4.3230
Epoch 1/10, Step: 249/540, total loss:6.6983
Epoch 1/10, Step: 299/540, total loss:5.8578
Epoch 1/10, Step: 349/540, total loss:6.6283
Epoch 1/10, Step: 399/540, total loss:6.4032
Epoch 1/10, Step: 449/540, total loss:5.9323
Epoch 1/10, Step: 499/540, total loss:7.5971
Epoch 1, accuracy: 0.9850, validation loss: 0.0058
Epoch 2/10, Step: 9/540, total loss:5.0577
Epoch 2/10, Step: 59/540, total loss:3.1802
Epoch 2/10, Step: 109/540, total loss:3.2457
Epoch 2/10, Step: 159/540, total loss:2.3691
Epoch 2/10, Step: 209/540, total loss:5.8829
Epoch 2/10, Step: 259/540, total loss:4.4065
Epoch 2/10, Step: 309/540, total loss:3.9642
Epoch 2/10, Step: 359/540, total loss:3.4601
Epoch 2/10, Step: 409/540, total loss:3.7005
Epoch 2/10, Step: 459/540, total loss:2.4082
Epoch 2/10, Step: 509/540, total loss:4.9357
Epoch 2, accuracy: 0.9817, validation loss: 0.0051
Epoch 3/10, Step: 19/540, total loss:3.0975
Epoch 3/10, Step: 69/540, total loss:3.5217
Epoch 3/10, Step: 119/540, total loss:2.2069
Epoch 3/10, Step: 169/540, total loss:2.1020
Epoch 3/10, Step: 219/540, total loss:2.2251
Epoch 3/10, Step: 269/540, total loss:1.9019
Epoch 3/10, Step: 319/540, total loss:5.7179
Epoch 3/10, Step: 369/540, total loss:2.3129
Epoch 3/10, Step: 419/540, total loss:2.5967
Epoch 3/10, Step: 469/540, total loss:3.6328
Epoch 3/10, Step: 519/540, total loss:3.5887
Epoch 3, accuracy: 0.9783, validation loss: 0.0083
Epoch 4/10, Step: 29/540, total loss:2.9215
Epoch 4/10, Step: 79/540, total loss:2.5634
Epoch 4/10, Step: 129/540, total loss:2.2718
Epoch 4/10, Step: 179/540, total loss:2.1903
Epoch 4/10, Step: 229/540, total loss:3.9422
Epoch 4/10, Step: 279/540, total loss:4.1438
Epoch 4/10, Step: 329/540, total loss:2.5862
Epoch 4/10, Step: 379/540, total loss:2.5173
Epoch 4/10, Step: 429/540, total loss:2.3017
Epoch 4/10, Step: 479/540, total loss:2.8343
Epoch 4/10, Step: 529/540, total loss:2.1349
Epoch 4, accuracy: 0.9817, validation loss: 0.0047
Epoch 5/10, Step: 39/540, total loss:1.4322
Epoch 5/10, Step: 89/540, total loss:0.5360
Epoch 5/10, Step: 139/540, total loss:1.1301
Epoch 5/10, Step: 189/540, total loss:1.0405
Epoch 5/10, Step: 239/540, total loss:1.5703
Epoch 5/10, Step: 289/540, total loss:1.2585
Epoch 5/10, Step: 339/540, total loss:2.3381
Epoch 5/10, Step: 389/540, total loss:1.5830
Epoch 5/10, Step: 439/540, total loss:1.9213
Epoch 5/10, Step: 489/540, total loss:2.2533
Epoch 5/10, Step: 539/540, total loss:2.3482
Epoch 5, accuracy: 0.9700, validation loss: 0.0080
Epoch 6/10, Step: 49/540, total loss:0.6322
Epoch 6/10, Step: 99/540, total loss:1.8027
Epoch 6/10, Step: 149/540, total loss:0.9355
Epoch 6/10, Step: 199/540, total loss:0.7137
Epoch 6/10, Step: 249/540, total loss:0.2431
Epoch 6/10, Step: 299/540, total loss:0.7320
Epoch 6/10, Step: 349/540, total loss:1.9910
Epoch 6/10, Step: 399/540, total loss:0.5313
Epoch 6/10, Step: 449/540, total loss:2.3987
Epoch 6/10, Step: 499/540, total loss:1.3163
Epoch 6, accuracy: 0.9767, validation loss: 0.0059
Epoch 7/10, Step: 9/540, total loss:2.2099
Epoch 7/10, Step: 59/540, total loss:1.0356
Epoch 7/10, Step: 109/540, total loss:0.3613
Epoch 7/10, Step: 159/540, total loss:0.3013
Epoch 7/10, Step: 209/540, total loss:0.9905
Epoch 7/10, Step: 259/540, total loss:0.9220
Epoch 7/10, Step: 309/540, total loss:1.8809
Epoch 7/10, Step: 359/540, total loss:0.6842
Epoch 7/10, Step: 409/540, total loss:0.8110
Epoch 7/10, Step: 459/540, total loss:0.5501
Epoch 7/10, Step: 509/540, total loss:0.1008
Epoch 7, accuracy: 0.9783, validation loss: 0.0114
Epoch 8/10, Step: 19/540, total loss:0.1257
Epoch 8/10, Step: 69/540, total loss:0.6099
Epoch 8/10, Step: 119/540, total loss:0.0731
Epoch 8/10, Step: 169/540, total loss:0.0690
Epoch 8/10, Step: 219/540, total loss:0.0463
Epoch 8/10, Step: 269/540, total loss:0.7255
Epoch 8/10, Step: 319/540, total loss:1.3453
Epoch 8/10, Step: 369/540, total loss:0.9634
Epoch 8/10, Step: 419/540, total loss:2.3351
Epoch 8/10, Step: 469/540, total loss:1.0744
Epoch 8/10, Step: 519/540, total loss:0.9177
Epoch 8, accuracy: 0.9833, validation loss: 0.0094
Epoch 9/10, Step: 29/540, total loss:0.0825
Epoch 9/10, Step: 79/540, total loss:0.5531
Epoch 9/10, Step: 129/540, total loss:0.1029
Epoch 9/10, Step: 179/540, total loss:1.6060
Epoch 9/10, Step: 229/540, total loss:1.4814
Epoch 9/10, Step: 279/540, total loss:0.2794
Epoch 9/10, Step: 329/540, total loss:0.8993
Epoch 9/10, Step: 379/540, total loss:0.2361
Epoch 9/10, Step: 429/540, total loss:0.8100
Epoch 9/10, Step: 479/540, total loss:1.0517
Epoch 9/10, Step: 529/540, total loss:1.9690
Epoch 9, accuracy: 0.9800, validation loss: 0.0081
Epoch 10/10, Step: 39/540, total loss:0.3611
Epoch 10/10, Step: 89/540, total loss:1.1712
Epoch 10/10, Step: 139/540, total loss:1.2629
Epoch 10/10, Step: 189/540, total loss:0.4406
Epoch 10/10, Step: 239/540, total loss:0.7798
Epoch 10/10, Step: 289/540, total loss:0.7707
Epoch 10/10, Step: 339/540, total loss:0.9512
Epoch 10/10, Step: 389/540, total loss:0.1460
Epoch 10/10, Step: 439/540, total loss:0.4900
Epoch 10/10, Step: 489/540, total loss:1.2117
Epoch 10/10, Step: 539/540, total loss:0.6272
Epoch 10, accuracy: 0.9783, validation loss: 0.0093
```

可以看到，第一轮的模型就是最优，准确率达到0.9850，使用该模型得到的分数是0.39905

## 调用ChatGLM的API的一点点尝试

看了群友的发言，感觉单纯采用从text中提取keywords已经无法提高准确率，于是转向使用生成式，that's to say，微调大模型

~~在deadline前，才想起来尝试使用大模型~~

困于本地环境不足，尝试调用API，这里为了保持稳定性选用了国产的ChatGLM-std版本的API，代码也敲好了（属于是第一个完完整整自己敲下来的东西），~~创业未半而中道崩殂~~，跑到第9个被官方ban掉了，原因是里面有敏感词，验证可行（**请不要输入易生成敏感用语的话题**）



> To be continued

## 更新日志

2023年7月21日：跑通了baseline

2023年7月22日：对baseline进行了fine-tune，实际结果略微优于baseline；尝试使用BERT模型，但是从运行中输出的部分参数来看，结果不尽如人意，同时显存爆了，建议上云

2023年7月23日：使用BERT模型进行训练，epoch=10，训练出的最优模型的准确率是0.9800，实际获得分数也优于22日的fine-tuned baseline

2023年7月24日：官方更新了B集，砍掉了Keywords，准确率下降不少，选用BERT模型的分数稍微高一点点

2023年7月26日：调用ChatGLM的API，但是由于触碰敏感词被ban掉了，所以可能需要本地的Chat GLM模型，但是huggingface上下载下来的是原始模型，自己无法完成训练，有训练完的模型可以使用，之后再进行尝试