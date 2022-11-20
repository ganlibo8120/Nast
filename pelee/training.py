from PIL import Image
import cv2 as cv
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import io as iomat
from skimage import io
import torch
import matplotlib.pyplot as plt
import DataLoader as dataset
import torch.optim as opt
import model
import Res3D

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(1)
def Evalution(encoder, test_root, epoch):
    encoder.eval()
    decoder.eval()
    data = dataset.Self_dataset(root_dir=test_root)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    sum = 0
    acc = 0
    for i, batch in enumerate(dataloader):
        x = batch['image']
        x = x.permute(0, 2, 1, 3, 4)
        #x = x[:, :, -2:-1, :, :]
        label = batch['label']
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
    encoder.train()
    decoder.train()
    return acc/sum



device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = dataset.Self_dataset(root_dir = 'dataset1')
dataloader = DataLoader(data, batch_size=32, shuffle= True)
encoder = model.Res3DV2().to(device)
decoder = model.VIT().to(device)
optimizer = opt.AdamW(encoder.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.02)
schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1)
loss_fuc = torch.nn.CrossEntropyLoss()
LOSS = []

warm_up = 5

encoder.train()
decoder.train()

for epoch in range(warm_up):

    optimizer.param_groups[0]['lr'] = 0.00001
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x = batch['image']
        x = x.permute(0, 2, 1, 3, 4)
        #x = x[:, :, -2:-1, :, :]bai'du

        label = batch['label']
        cls = encoder(x.to(device))
        #pred = decoder(x,)
       # print(pred.shape,label.shape)
        loss = loss_fuc(cls, label.to(device))
        loss.backward()
        optimizer.step()
        print('%10f'%loss.item())


print('#####################################')

best_acc = 0

for epoch in range(100):
    if epoch % 3 == 1 and epoch >10:
        schedule.step()
        #optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.95
        acc = Evalution(encoder,'test',epoch)
        print('epoch:', epoch, ' acc:', acc, ' best:', best_acc)

        if acc> best_acc:
            best_acc = acc
           # torch.save(encoder.state_dict(),'best.pt')
    if epoch %20 ==19:
        torch.save(encoder.state_dict(), 'discriminator'+str(epoch).zfill(4)+'.pt')

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x = batch['image']
        x = x.permute(0, 2, 1, 3, 4)
        #x = x[:,:,-2:-1,:,:]
        label = batch['label']
        cls = encoder(x.to(device))
        #pred = decoder(x,tg)
       # print(pred.shape,label.shape)
        loss = loss_fuc(cls, label.to(device))
        loss.backward()
        optimizer.step()
        #print('%10f'%loss.item())
        LOSS.append(loss.item())
plt.plot(LOSS)
plt.show()



