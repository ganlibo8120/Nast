
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
torch.manual_seed(999)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evalution(model):
    root = 'C:/Fashionmnist'  # 'C:/MNIST/testing_data'

    PicTransform = transforms.Compose([transforms.ToTensor()])
    #
    # 用于批量读取图片，并对图像进行预处理
    train_data = FashionMNIST(root=root, train=False, transform=PicTransform)
    train_loader = Data.DataLoader(dataset=train_data,
                                   shuffle=True,
                                   batch_size=16)
    model.eval()
    right = 0
    sum = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = repeat(batch_x, 'n () h w -> n cc h w', cc=3)
            predict = model(batch_x.to(device))
            h = batch_y.shape
            h = int(h[0])
            sum = sum + h
            predict = torch.argmax(predict, dim=1)

            # print(predict.shape)

            predict = predict.cpu().data.numpy()
            batch_y = batch_y.numpy()
            bol = predict == batch_y
            # print(predict,batch_y)

            # print(predict)
            # print(bol)
            # print(batch_y)
            # print(np.sum(bol) / 16)
            right = right + np.sum(bol)


    return right / sum

def evalution0(model):
    root = 'C:/Fashionmnist'  # 'C:/MNIST/testing_data'

    PicTransform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize(std=1,mean=0)])
    #
    # 用于批量读取图片，并对图像进行预处理
    train_data = FashionMNIST(root=root, train=False, transform=PicTransform)
    train_loader = Data.DataLoader(dataset=train_data,
                                   shuffle=True,
                                   batch_size=16)
    model.eval()
    right = 0
    sum = 0
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = repeat(batch_x, 'n () h w -> n cc h w', cc=3)
            # print(batch_x.shape)
            predict = model(batch_x.to(device))
            h = batch_y.shape
            h = int(h[0])
            sum = sum + h
            predict = torch.argmax(predict, dim=1)

            # print(predict.shape)

            predict = predict.cpu().data.numpy()
            batch_y = batch_y.numpy()
            bol = predict == batch_y
            # print(predict,batch_y)

            # print(predict)
            # print(bol)
            # print(batch_y)
            # print(np.sum(bol) / 16)
            right = right + np.sum(bol)


    return right / sum

