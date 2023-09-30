import torch
import torchvision
import matplotlib.pyplot as plt


trans=torchvision.transforms.ToTensor()#将图像数据改为32位浮点数

train_data=torchvision.datasets.MNIST(
    root="./data",train=True,transform=trans,download=True)

test_data=torchvision.datasets.MNIST(
    root="./data",train=False,transform=trans,download=True)
