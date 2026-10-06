import torch
from torch import nn
from torch.nn import functional as F

def block(i,o):return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.ReLU())
class ChangeNet(nn.Module):
    def __init__(self,siamese=False,width=8):
        super().__init__();self.siamese=siamese
        self.e1=block(4 if siamese else 8,width);self.e2=block(width,2*width);self.e3=block(2*width,4*width)
        self.d2=block(6*width,2*width);self.d1=block(3*width,width);self.out=nn.Conv2d(width,1,1)
    def encode(self,x):
        a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2));return a,b,c
    def forward(self,x):
        if self.siamese:
            p=self.encode(x[:,:4]);q=self.encode(x[:,4:]);a,b,c=[(i-j).abs() for i,j in zip(p,q)]
        else:a,b,c=self.encode(x)
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
