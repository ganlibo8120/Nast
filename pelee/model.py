import torch.nn as nn
from torch.nn import  init
import  torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from  torchsummary  import  summary
import torchvision
#model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=1024)
#out = model(torch.rand([10, 32, 512]),torch.rand([1, 32, 512]))
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Resnet(nn.Module):
    def __init__(self, num_class=2,):
        super(Resnet, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(512,num_class)
    def forward(self, x):
        n, b, c, h, w = x.shape
        fc = []
        feature = []
        for i in range(n):
            tep = x[i, :]
            tep = self.backbone.conv1(tep)
            tep = self.backbone.bn1(tep)
            tep = self.backbone.relu(tep)
            tep = self.backbone.layer2(tep)
            tep = self.backbone.layer3(tep)
            tep = self.backbone.layer4(tep)

            feature.append(tep.view(tep.size(0),tep.size(1), -1).permute(2,0,1))
            tep = self.backbone.avgpool(tep)
            tep = tep.view(tep.size(0), tep.size(1))
            fc.append(tep.unsqueeze(0))
            #tep = self.fc(tep)
            #tep = tep.unsqueeze(0)
            #fc.append(tep)

        fc = torch.cat(fc, dim=0)
        fc = fc.mean(dim=0)
        #
        feature = torch.cat(feature, dim=0)
        #print(feature.shape, fc.shape)
        clss = self.fc(fc)


        #feature = feature.view(feature.size(0), feature.size(1), feature.size(2), -1)
        #feature=feature.permute(0, 3, 1, 2)
        #feature = feature.view(feature.size(0)*feature.size(1), feature.size(2), feature.size(3))

        #print(fc.shape,feature.shape)
        #print(clss.shape)
        #return feature,fc.unsqueeze(0),clss
        return clss

class TransEncoder(nn.Module):
    def __init__(self,nhead = 8, num_class=2,num_encoder_layers = 1,dim_feedforward = 1024,d_model=512,imageSize = [56,56], patch_height = 14, patch_width = 14, channel = 3):
        super(TransEncoder, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self.patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = patch_height,p2 =patch_width),
            nn.Linear(channel*patch_width*patch_height,d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,dim_feedforward=dim_feedforward,nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=num_encoder_layers)
        self.emdedding = nn.Parameter(torch.randn(1,imageSize[0]*imageSize[1]//patch_height//patch_width+1,d_model))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(d_model,1024),
            nn.Dropout(0.2),
            nn.Linear(1024, num_class),
        )

    def forward(self,x):
        n,b,c,h,w = x.shape
        feature = []
        cls = []
        for i in range(n):
            tep = x[i, :]

            tep = self.patch(tep)
            #print(tep.shape)
            cls_token = repeat(self.cls_token, '() n d-> b n d', b=b)
            tep = torch.cat([tep, cls_token], dim=1)
            tep = tep.permute(1, 0, 2)
            #print(tep.shape)
            tep = self.encoder(tep)
            feature.append(tep[0:-1, :, :])
            cls.append(tep[-1:, :, :])

        feature = torch.cat(feature, dim=0)
        cls = torch.cat(cls, dim=0)
        cls = torch.mean(cls, dim=0)
        cls = cls.view(1, cls.size(0), cls.size(1))
        #print(feature.shape,cls.shape)


        #分类
        cls = self.fc(cls)
        cls = cls.view(cls.size(1),-1)
        #分类
        return  cls






class VIT(nn.Module):
    def __init__(self, d_model = 512,num_class=2, nhead = 2, num_encoder_layers = 1, num_decoder_layers =2  ,dim_feedforward = 1024):
        super(VIT, self).__init__()
        self.model = nn.Transformer(d_model=d_model,nhead = nhead,num_decoder_layers=num_decoder_layers,
                                    num_encoder_layers= num_encoder_layers,dim_feedforward=dim_feedforward)

        self.position_code = None
        self.fc = nn.Linear(d_model,num_class)

    def forward(self, x, target):
        if self.position_code is not None:
            x = x + self.position_code
        out = self.model(x, target)
        out =self.fc(out)
        out =out.view(out.size(1),-1)
        return out

#print(torchvision.models.resnet18())
class ResLayers(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ResLayers, self).__init__()
        self.down_conv_a=nn.Sequential(
            nn.Conv3d(in_channels=inchannel,out_channels=outchannel//2,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)),
            nn.BatchNorm3d(outchannel//2),
            nn.ReLU(),
            nn.Conv3d(in_channels=outchannel//2,out_channels=outchannel,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
            nn.BatchNorm3d(outchannel),
        )
        self.down_conv_b = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1, 1), stride=(1, 2, 2),
                      padding=(0, 0, 0)),
            nn.BatchNorm3d(outchannel),
        )
        self.add_relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=outchannel, out_channels=outchannel//2, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(outchannel//2),
            nn.ReLU(),
            nn.Conv3d(in_channels=outchannel//2, out_channels=outchannel, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(outchannel),
        )
        self.add_relu2 = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.down_conv_a(x) + self.down_conv_b(x)
        x = self.add_relu1(x)
        x = x + self.conv(x)
        return self.add_relu2(x)


class ResLayers2d(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ResLayers2d, self).__init__()
        self.down_conv_a=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel//2,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=outchannel//2,out_channels=outchannel,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(outchannel),
        )
        self.down_conv_b = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1), stride=(2, 2),
                      padding=(0, 0)),
            nn.BatchNorm2d(outchannel),
        )
        self.add_relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel//2, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=outchannel//2, out_channels=outchannel, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.BatchNorm2d(outchannel),
        )
        self.add_relu2 = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.down_conv_a(x) + self.down_conv_b(x)
        x = self.add_relu1(x)
        x = x + self.conv(x)
        return self.add_relu2(x)

class ResLayers3DBottleNeck(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ResLayers3DBottleNeck, self).__init__()

        self.down_conv_a=nn.Sequential(
            nn.Conv3d(in_channels=inchannel,out_channels=outchannel//2,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)),
            nn.BatchNorm3d(outchannel//2),
            nn.ReLU(),
            nn.Conv3d(in_channels=outchannel//2,out_channels=outchannel,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
            nn.BatchNorm3d(outchannel),
        )
        self.down_conv_b = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=(1, 1, 1), stride=(1, 2, 2),
                      padding=(0, 0, 0)),
            nn.BatchNorm3d(outchannel),
        )
        self.add_relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=outchannel, out_channels=outchannel//2, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(outchannel//2),
            nn.ReLU(),
            nn.Conv3d(in_channels=outchannel//2, out_channels=outchannel, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(outchannel),
        )
        self.add_relu2 = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.down_conv_a(x) + self.down_conv_b(x)
        x = self.add_relu1(x)
        x = x + self.conv(x)
        return self.add_relu2(x)




class Res3D(nn.Module):
    def __init__(self,inchannel=3,basechannel=64, num_class=2):
        super(Res3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=basechannel, kernel_size=(1, 5, 5), stride=(1, 1, 1),
                      padding=(0, 2, 2)),
            nn.BatchNorm3d(basechannel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.res2 = nn.Sequential(
            ResLayers(basechannel, 128),
            nn.Dropout(0.1)
        )
        #self.CoAt2= CoAten3D(dim=128,n=4)
        self.res3 = nn.Sequential(
            ResLayers(128, 256),
            nn.Dropout(0.1),
        )
        #self.CoAt3 = CoAten3D(dim=256, n=8)
        self.res4 = nn.Sequential(
            ResLayers(256, 512),
            nn.Dropout(0.1)
        )

        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.0),
            nn.Linear(256, num_class)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        #x = self.CoAt2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        return x




class Res2D(nn.Module):
    def __init__(self,inchannel=3,basechannel=64, num_class=2):
        super(Res2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=basechannel, kernel_size=(5, 5), stride=(1, 1),
                      padding=(2, 2)),
            nn.BatchNorm2d(basechannel),
            nn.ReLU(),
            nn.Dropout(0.0)
        )
        self.res2 = nn.Sequential(
            ResLayers2d(basechannel, 128),
            nn.Dropout(0.0)
        )
        #self.CoAt2= CoAten3D(dim=128,n=4)
        self.res3 = nn.Sequential(
            ResLayers2d(128, 256),
            nn.Dropout(0.0),
        )
        #self.CoAt3 = CoAten3D(dim=256, n=8)
        self.res4 = nn.Sequential(
            ResLayers2d(256, 512),
            nn.Dropout(0.0)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.0),
            nn.Linear(256, num_class)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        #x = self.CoAt2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.head(x)
        return x


class Res3DV2(nn.Module):
    """
    stem 和  Resblock with bottleneck
    """
    def __init__(self,inchannel=3,basechannel=64,num_class=2):
        super(Res3DV2, self).__init__()
        self.stem= nn.Sequential(
            nn.Identity()
        )

        self.stem1 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel,out_channels=basechannel//4,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
            nn.BatchNorm3d(basechannel//4),
            nn.ReLU(),
            nn.Conv3d(in_channels=basechannel//4,out_channels=basechannel//2,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel,out_channels=basechannel//2,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))
        )
        self.stem_act = nn.Sequential(
            nn.BatchNorm3d(basechannel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.res2 = nn.Sequential(
            ResLayers3DBottleNeck(basechannel, 128),
            nn.Dropout(0.1)
        )
        # self.CoAt2= CoAten3D(dim=128,n=4)
        self.res3 = nn.Sequential(
            ResLayers3DBottleNeck(128, 256),
            nn.Dropout(0.1),
        )
        # self.CoAt3 = CoAten3D(dim=256, n=8)
        self.res4 = nn.Sequential(
            ResLayers3DBottleNeck(256, 512),
            nn.Dropout(0.1)
        )

        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.0),
            nn.Linear(256, num_class)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        #x = self.conv1(x)
        x = torch.cat([self.stem1(x),self.stem2(x)],dim=1)
        x = self.stem(x)
        x = self.res2(x)
        # x = self.CoAt2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class CoAten3D(nn.Module):
    def __init__(self,dim,n):
        super(CoAten3D, self).__init__()
        self.x =nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim//n, kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0),bias=False),
            nn.Conv3d(in_channels=dim//n, out_channels=dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.GroupNorm(n, dim),
            nn.Sigmoid(),
        )

        self.y = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim // n, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), bias=False),
            nn.Conv3d(in_channels=dim // n, out_channels=dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), bias=False),
            nn.GroupNorm(n, dim),
            nn.Sigmoid(),
        )
        self.z = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim // n, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), bias=False),
            nn.Conv3d(in_channels=dim // n, out_channels=dim, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                      padding=(0, 0, 0), bias=False),
            nn.GroupNorm(n, dim),
            nn.Sigmoid(),
        )
    def forward(self,input):
        b,c,t,h,w = input.shape
        x = nn.AdaptiveAvgPool3d((1,1,w))(input)
        y = nn.AdaptiveAvgPool3d((1,h,1))(input)
        z = nn.AdaptiveAvgPool3d((t,1,1))(input)
        x = self.x(x)
        y = self.y(y)
        z = self.z(z)
        output = input * x * y* z
        return output




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((1,None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1,1, None))
        self.pool_t = nn.AdaptiveAvgPool3d((None,1,1))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, t, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)


        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

'''Net = Res3D().to("cuda:0")

summary(Net,(3,8,56,56))'''
if __name__=="__main__":
    net = Res3DV2().to("cuda")
    ins = torch.randn([3, 224, 224]).to("cuda")
    summary(net,(3,8,224,224))
