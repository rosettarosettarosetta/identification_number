import torch
import torchvision
import matplotlib.pyplot as plt

#参数
BATCH_SIZE = 50
LR = 0.001  # 学习率

#数据部分
trans=torchvision.transforms.ToTensor()#将图像数据改为32位浮点数

train_data=torchvision.datasets.MNIST(
    root="./data",train=True,transform=trans,download=True)

test_data=torchvision.datasets.MNIST(
    root="./data",train=False,transform=trans,download=True)

print("data set exists\n"+"training data: "+str(len(train_data)))
print("test data: "+str(len(test_data))+"\n")


torch.manual_seed(125)   # 使用随机化种子使神经网络的初始化每次都相同

print("pic size: "+str(test_data[0][0].shape))#前一个0是指第几个图片 后一个是指图片本身

train_loader = torch.utils.data.DataLoader(   #数据加载器
    dataset=train_data,
    batch_size=BATCH_SIZE,    #样本数量
    shuffle=True  # 是否打乱数据
)