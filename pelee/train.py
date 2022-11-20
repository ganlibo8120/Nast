import cv2 as cv
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.models as model
import torchvision.transforms as transforms
from   torchvision.datasets import ImageFolder
import torch.utils.data as Data
import torch.optim as opt
import matplotlib.pyplot as plt
import time
from  torchvision.datasets import FashionMNIST
from einops import rearrange, repeat
from evalution import evalution,evalution0
torch.manual_seed(999)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),  # 卷积的大小我自己改了一下 前两个卷积后面居然没有归一化层，这与Matlab给的标准网络不同
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # 自己加的
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),  # 自己加的
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 缺省默认50%抛弃
            nn.Linear(256 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.classifier(x)
        return x

class SE(nn.Module):
    def __init__(self, d):
        super(SE, self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc1 = nn.Linear(d, round(d/8))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(round(d/8), d)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x2 = x
        #print(x.shape)
        x = self.avepool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmod(x)
        #print(x1.requires_grad)
        #x = torch.unsqueeze(x, 2)
        #x = torch.unsqueeze(x, 3)
        #print(x.shape)
        x_ = x.transpose(0, 1)
        length=x_.size(0)*x_.size(1)
        x_=x_.reshape(1,length)

        x_=x_.repeat(x1.size(2),x1.size(3))
        x_=x_.reshape(x1.size(2),x1.size(3),x1.size(1),x1.size(0))
        x_ = x_.permute(3, 2, 0, 1)
        #print(x_.shape)
        #for ii in range(x1.size(2)):
         #  for j in range(x1.size(3)):
       # print(x)
        #print(x_)
        x2 = torch.einsum('ijkl,ijkl->ijkl', [x1, x_])
        #print(x1.requires_grad)
        #x1 = x1.contiguous()
        x2=x2+x1
        return x2


class attention(nn.Module):
    def __init__(self, d):
        super(attention, self).__init__()
        self.conv1 = nn.Conv2d(d, round(d/2), 1, 1, 0)
        self.conv2 = nn.Conv2d(d, round(d/2), 1, 1, 0)
        self.conv3 = nn.Conv2d(d, round(d/2), 1, 1, 0)
        self.conv4 = nn.Conv2d(round(d/2), d, 1, 1, 0)
        self.soft = nn.Softmax(0)

    def forward(self, x):
        h= x.shape
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        #print(x1.shape)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 3, 1)
        x3 = torch.flatten(x3, start_dim=0, end_dim=2)
        x1 = torch.flatten(x1, start_dim=0, end_dim=2)
        x2 = torch.flatten(x2, start_dim=0, end_dim=2)
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.transpose(x2, 0, 1)
        # print(x2.shape)
        x12 = torch.matmul(x1,x2)

        x12 = self.soft(x12)
        x23 = torch.matmul(x12, x3)
        # print(x23.shape)
        x23 = torch.reshape(x23,[h[0],h[2],h[3],round(h[1]/2)])
        # print(x23.shape)
        x23 = x23.permute(0, 3, 1, 2)
        x23 = self.conv4(x23)
        out = x23 + x
        return out
# block=attention(512)
#,1024)
# block.eval()
# x = torch.rand(3, 512, 10, 10)

#predictions =block(x)
#print(predictions.shape)


Loss = []
il = []

class self_define_net(nn.Module):
    def __init__(self,num_classes=10):
        super(self_define_net, self).__init__()
        self.backbone = model.resnet18(pretrained=True)
        self.fc = nn.Linear(512,num_classes,bias=True)
        self.SE1 = SE(64)
        self.SE2 = SE(128)
        self.SE3 = SE(256)
        self.SE4 = SE(512)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.SE3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#自定义CNN分类网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
           # SE(8),
            nn.Conv2d(8, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
          #  attention(16)
            # SE(16)
        )
        #self.conv4 = attention(64)
        self.conv3 = nn.Sequential(
            #nn.Conv2d(32, 64, 2, 1, 0),
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
          #  SE(32)
            #nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(
          #  nn.Dropout(0.1),
            nn.Linear(288, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        #x = self.conv4(x)
        x = self.conv3(x)
      #  print(x.shape)
       # print(x.shape)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)

        output = self.out(x)
        #print(output.shape)
        return output


EPOCH = 1
BATCH_SIZE = 4
LR = 0.001
DOWNLOAD_MNIST = False

# 获取数据
root = 'C:/Fashionmnist'#'C:/MNIST/training_data'

PicTransform = transforms.Compose([transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomCrop(24)]),transforms.Resize(224),transforms.ToTensor(),
                                   transforms.Normalize(std=1,mean=0)])
#
# 用于批量读取图片，并对图像进行预处理
train_data = FashionMNIST(root=root, train=True,transform=PicTransform)
train_loader = Data.DataLoader(dataset=train_data,
                               shuffle=True,
                               batch_size=64)

model =  self_define_net().to(device)#model.resnet18(num_classes=10).to(device)#Net().to(device)

# 使用损失函数和优化器

loss_fuc = nn.CrossEntropyLoss()
optimizer = opt.AdamW(model.parameters(), lr=0.001, betas=(0.9,0.95),weight_decay=0.05)
i = 0
model.train()
time_start=time.time()
for epoch in range(0,80):
    if epoch>0:
       optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.95#0.001*np.cos(np.pi/2*epoch/80)
    for step, (batch_x, batch_y) in enumerate(train_loader):
        #print(batch_x.shape)
        batch_x = repeat(batch_x, 'n () h w -> n cc h w', cc=3)
        predict = model(batch_x.to(device))

        loss = loss_fuc(predict, batch_y.to(device))
        # loss = model(batch_x, batch_y)

        i = i + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
        il.append(i)

        #print("epoch=%s,step=%s,loss=%s"%(epoch,i,loss.item()))
    print('epoch=',epoch,'acc=',evalution0(model))


time_end=time.time()
print('totally cost',time_end-time_start)

fig = plt.figure(2)
# ax = fig.subplot(111)
plt.plot(il, Loss)
plt.show()
torch.save(model, 'classify_model.pkl')

