import cv2
from PIL import Image
import numpy as np
#引入torch，torchvision库
import torch
from torchvision import datasets,transforms
#datasets是加载图片数据的方法，transforms是图像预处理的方法
from torch import nn,optim
#nn即neural network，是专门为神经网络设计的模块化接口；optim是优化器
import torch.nn.functional as F

#构建LeNet模型
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
    def forward(self,x):
    #池化，池化核2x2
        x = F.max_pool2d(F.relu(self.c1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.c3(x)),(2,2))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=s
        return num_features
      
      
#自己手写数字来进行测试
#读取训练好的模型
lenet = LeNet()
lenet.load_state_dict(torch.load('model.pkl'))
#输入图片地址来读-取图片。返回torch.tensor
def image_loader(image_name):
    image = cv2.imread(image_name)#读取图片
    #转换图片为灰度图片
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #高斯滤波
    gauss_img = cv2.GaussianBlur(gray_image,(5,5),0,0,cv2.BORDER_DEFAULT)
    #修改图片大小为28*28像素
    image = cv2.resize(gauss_img,(28,28))

    for h in range(28):
        for w in range(28):
            image[h][w] = 255 - image[h][w]
    to_pil_img = transforms.ToPILImage()#tensor 重新转化成图片格式
    image1 = to_pil_img(image)
    image1.show()
    #归一化像素数据
    image = transform(image)
    return torch.unsqueeze(image.to(torch.float),0)


image = image_loader('test_9.jpg')
print(image.size())
print(lenet(image))
#tensor 重新转化成图片格式
image = torch.squeeze(image,0)
to_pil_img = transforms.ToPILImage()#tensor 重新转化成图片格式
image = to_pil_img(image)
#image.show()
