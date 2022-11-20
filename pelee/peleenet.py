import torch
import torch.nn.functional as F
import  torch.nn as nn
from  torchsummary import summary
import  numpy as np
class Block(nn.Module):
    def __init__(self,in_channel,out_channel,s,g=2):
        super(Block, self).__init__()
        self.conv = nn.Parameter(torch.randn(((in_channel+ (out_channel-1)*s)//1,1,1)))
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.s = s
        self.base = torch.tensor(list(range(in_channel))).cuda()
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x ):
        b,c,h,w =x.shape
        #print(c,self.in_channel)

        #kernel = []
        kernel = torch.zeros([self.out_channel, self.in_channel,1,1],device='cuda')
        #print(self.in_channel,self.out_channel)
        base = self.base#np.array(range(self.in_channel))
        for i in range(self.out_channel):
            kernel[i, :, :, :] = self.conv[base+self.s*i, :, :]
            #kernel.append(tmp)
            #print(self.conv[:,self.base,:,:].shape,tmp.shape)
        #print(kernel.shape)
        out = F.conv2d(x, kernel)
        return self.relu(self.norm(out))


if __name__ == '__main__':
    x = torch.randn([2,128,32,32],requires_grad=False).cuda()

    normal_conv = nn.Sequential(nn.Conv2d(128,128,1,1),

                                ).cuda().eval()

    SD_conv  = nn.Sequential(Block(128,128,32),
                            ).cuda().eval()
    summary(SD_conv,(128,32,32))
    import  time

    for i in range(10):
        SD_conv(x)
    t = time.time()
    for i in range(100):
        SD_conv(x)
    print((time.time() - t)/100)

