import model as model
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import io as iomat
from skimage import io
import torch
import matplotlib.pyplot as plt
import DataLoader as dataset
import torch.optim as opt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def Evalution(encoder,test_root,epoch,sample_num=8):

    encoder.eval()
    data = dataset.Self_dataset(root_dir=test_root)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    sum = 0
    acc = 0
    for i, batch in enumerate(dataloader):
        x = batch['image']
        #print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        label = batch['label']
        x = x[:,:,sample_num:8,:,:]
        pred = encoder(x.to(device))
        #pred = decoder(x, tg)
        pred = torch.argmax(pred,dim=1)
        #print(pred.shape)
        pred = pred.cpu().detach().numpy()
        label = label.cpu().numpy()
        sum = sum+ label.shape[0]
        #print(i,' ', pred)
        bol = pred == label
        acc = acc + np.sum(bol)
    #encoder.train()
    return acc/sum



net = model.Res3D()
net.load_state_dict(torch.load("best.pt"))
net.to(device)
print(Evalution(net,"test",1,3))
