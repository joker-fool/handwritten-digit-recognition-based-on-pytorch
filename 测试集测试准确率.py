from PIL import Image
import numpy as np
#引入torch，torchvision库
import torch
from torchvision import datasets,transforms
#datasets是加载图片数据的方法，transforms是图像预处理的方法
from torch import nn,optim
#nn即neural network，是专门为神经网络设计的模块化接口；optim是优化器
import torch.nn.functional as F

#加载测试集
testset = datasets.MNIST('data', train=False, download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers = 2)
#测试训练完的模型
def test(testloader,model):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy:%d %%' % (100*correct/total))

lenet.load_state_dict(torch.load('model.pkl'))
test(testloader,lenet)

