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
        self.conv1=nn.Conv2d(3,10,3) #其中1代表灰度图片的通道 10：输出通道，5：卷积核
        self.conv2=nn.Conv2d(10,20,1) #10:输入通道 20：输出通道 3：卷积核大小
        #在添加一个全连接层
        self.fc1=nn.Linear(20*10*10,500)      #20*10*10:输入通道 500：输出通道
        self.fc2=nn.Linear(500,10)            #500：输入通道  10：输出通道

    def forward(self,x):
        input_size=x.size(0)    #batch_size
        x=self.conv1(x) #输入:batch*3*28*28, 输出:batch*10*24*24
        x=F.relu(x)     #保持shape不变
        x=F.max_pool2d(x,3,3)    #输入：batch*10*24*24 输出:batch*10*12*12

        x=self.conv2(x)           #输入：batch*12*12   输出：batch*20*10*10
        x=F.relu(x)
        x=x.view(input_size,-1) #拉伸，即降维一维数组 20*10*10=2000

        x=self.fc1(x) #输入:batch*2000 输出:batch*500
        x=F.relu(x) #保持shape不变
        x=self.fc2(x) #输入：batch*500 输出：batch*10
        output=F.log_softmax(x,dim=1)    #计算分类后每个数字的分类值
        return output

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

#残差块的定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
    model = ResNet(ResidualBlock, [2, 2, 2]).to('cuda')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir+'/pth/model.pth', map_location='cpu'))
    #torch.load(model.state_dict(), 'pth/model.pth')
    #torch.load_state_dict('pth/model.pth')
    return model
