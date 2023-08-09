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

# 将整体数据拆分为 30 份，放入数据加载器，就是一次性会有 <总样本数 / 30> 份数据同时并行计算，份数越多，并行越多，显存占用越大，需要根据需求来选择
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
    if accuracy>best_accuracy:
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

def infer(string: str) -> int:
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



