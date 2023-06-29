# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os
import torch.nn.functional as F

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        #先添加2个卷积层
        self.conv1=nn.Conv2d(1,10,5) #其中1代表灰度图片的通道 10：输出通道，5：卷积核
        self.conv2=nn.Conv2d(10,20,3) #10:输入通道 20：输出通道 3：卷积核大小
        #在添加一个全连接层
        self.fc1=nn.Linear(20*10*10,500)      #20*10*10:输入通道 500：输出通道
        self.fc2=nn.Linear(500,10)            #500：输入通道  10：输出通道
    def forward(self,x):
        input_size=x.size(0)    #batch_size
        x=self.conv1(x) #输入:batch*1*28*28, 输出:batch*10*24*24
        x=F.relu(x)     #保持shape不变
        x=F.max_pool2d(x,2,2)    #输入：batch*10*24*24 输出:batch*10*12*12

        x=self.conv2(x)           #输入：batch*12*12   输出：batch*20*10*10
        x=F.relu(x)
        x=x.view(input_size,-1) #拉伸，即降维一维数组 20*10*10=2000

        x=self.fc1(x) #输入:batch*2000 输出:batch*500
        x=F.relu(x) #保持shape不变
        x=self.fc2(x) #输入：batch*500 输出：batch*10
        output=F.log_softmax(x,dim=1)    #计算分类后每个数字的分类值
        return output


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def main():
    model = Digit()  # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir+'/pth/model.pth', map_location='cpu'))
    #torch.load(model.state_dict(), 'pth/model.pth')
    #torch.load_state_dict('pth/model.pth')
    model.to('cpu')
    return model


