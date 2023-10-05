import torch
import torchvision
import visualize
import matplotlib.pyplot as plt
import IPython
import algorithm

#参数
BATCH_SIZE = 50
EPOCHS=5
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

train_loader = torch.utils.data.DataLoader(   #数据加载器/迭代器iterator
    dataset=train_data,
    batch_size=BATCH_SIZE,    #样本数量
    shuffle=True,  # 是否打乱数据
    num_workers=2   #数据加载的线程数
)

#可视化
X, y = next(iter(torch.utils.data.DataLoader(train_data, batch_size=18)))
y_titles = [str(label.item()) for label in y]  # 将 y 转换为字符串列表
visualize.show_images(X.reshape(18, 28, 28), 2, 9, titles=y_titles)
plt.show()


#softmax
num_inputs=test_data[0][0].shape[1] * test_data[0][0].shape[2]
num_outputs=10 #类别量

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  #初始值 ，标准差，大小，进行梯度计算
b = torch.zeros(num_outputs, requires_grad=True)  #初始值为0