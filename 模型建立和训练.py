#引入必要的库和函数
from PIL import Image
import numpy as np
#引入torch，torchvision库
import torch
from torchvision import datasets,transforms
#datasets是加载图片数据的方法，transforms是图像预处理的方法
from torch import nn,optim
#nn即neural network，是专门为神经网络设计的模块化接口；optim是优化器
import torch.nn.functional as F

#ToTensor（）是将数据转化为Tensor对象，Normalize（）是对数据进行归一化
transform = transforms.Compose([  transforms.ToTensor(),  transforms.Normalize((0.1307,),(0.3081,))])

#使用datasets.MNIST（）来分别下载训练数据集和测试数据集
trainset = datasets.MNIST('data', train=True, download=False, transform=transform)
testset = datasets.MNIST('data', train=False, download=False, transform=transform)

#构建LeNet模型，用来识别手写数字
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        #卷积层c1，c3
        self.c1 = nn.Conv2d(1,6,(5,5))
        self.c3 = nn.Conv2d(6,16,(5,5))
        #全连接层fc1，2，3
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    #前向传播
    def forward(self,x):
    #池化，池化核2x2
    #输入1×28×28，c1卷积后为6×24×24，池化后为6×12×12
        x = F.max_pool2d(F.relu(self.c1(x)),(2,2))
    #c2卷积后为16×8×8，池化后为16×4×4
        x = F.max_pool2d(F.relu(self.c3(x)),(2,2))
    #view函数后，x形状从batch×16×4×4变为batch×256
        x = x.view(-1,self.num_flat_features(x))
    #全连接层1，256->120
        x = F.relu(self.fc1(x))
    #全连接层2，120->84
        x = F.relu(self.fc2(x))
    #全连接层3，84->10
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
      
#初始化LeNet模型
CUDA = 0
if CUDA:
    lenet = LeNet().cuda()
else:
    lenet = LeNet()
    
#损失函数为交叉熵函数,优化器为随机梯度下降
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lenet.parameters(),lr=0.001,momentum=0.9)      

#batch_size表示一次加载的数据量，shuffle = True表示遍历不同批次的数据时打乱顺序，num_workers=2表示使用两个子进程加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle = True, num_workers=2)
for i,data in enumerate(trainloader,0):
    inputs,labels = data
    print(labels)
    if i == 1:
        break
        
#训练模型，model为传入的模型，criterion为损失函数，optimizer为优化器，epochs为训练轮数
def train(model,criterion,optimizer,epochs=1):
    for epoch in range(epochs):
        running_loss = 0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            if CUDA:
                inputs,labels = inputs.cuda(),labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #反向传播
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%1000==999:
                print('[Epoch:%d,Batch:%5d] Loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                running_loss = 0.0

train(lenet,criterion,optimizer,epochs =1)
#保存训练好的模型，命名model.pkl
torch.save(lenet.state_dict(), 'model.pkl')
