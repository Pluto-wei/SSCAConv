import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict



class SSCAConv(nn.Module):
    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(SSCAConv, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

        self.attention=nn.Sequential(
            nn.Conv2d(in_planes,in_planes*(kernel_size**2),kernel_size,stride,padding,dilation,groups=in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes*(kernel_size**2),in_planes*(kernel_size**2),1,1,0,groups=in_planes),
            nn.Tanh()
        ) # b,1,H,W 全通道像素级通道注意力

        if use_bias==True:
            conv1 = nn.Conv2d(in_planes, in_planes, 1,1,0)
            self.bias=conv1.bias

    def forward(self,x):
        (b, n, H, W) = x.shape
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)

        atw=self.attention(x).reshape(b,n,k*k,n_H,n_W) #b,n,k*k,n_H*n_W
        unf_x=F.unfold(x,kernel_size=k,dilation=self.dilation,padding=self.padding,stride=self.stride).reshape(b,n,k*k,n_H,n_W) #b,n*k*k,n_H*n_W
        unf_y=unf_x*atw #b,n,k*k,n_H,n_W
        y=torch.sum(unf_y,dim=2,keepdim=False)#b,n,n_H,n_W

        if self.use_bias==True:
            y = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(y) + y

        return y



class Residual_Block(nn.Module):
    def __init__(self, channels, isd):
        super(Residual_Block, self).__init__()
        if isd==True:
            self.conv = nn.Sequential(
                SSCAConv(channels,3,1,1),
                nn.Conv2d(channels,channels,1,1,0),
                nn.ReLU(inplace=True),
                SSCAConv(channels,3,1,1),
                nn.Conv2d(channels, channels, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
                nn.Conv2d(channels, channels, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1,groups=channels),
                nn.Conv2d(channels, channels, 1, 1, 0),
            )

    def forward(self, x):
        return self.conv(x) + x


class SSCANet(nn.Module):
    def __init__(self,channels,block_num,isd=True,inchannel=8,in2 = 1):
        super(SSCANet, self).__init__()
        self.head_conv=nn.Sequential(
            nn.Conv2d(inchannel+in2,channels,3,1,1),
            nn.ReLU(inplace=True)
        )
        self.rbs=self._make_resblocks(channels,block_num,isd)
        self.tail_conv=nn.Conv2d(channels,inchannel,3,1,1)

    def forward(self, pan, lms):
        x = torch.cat([pan, lms], dim=1)
        x = self.head_conv(x)
        x = self.rbs(x)
        x = self.tail_conv(x)
        return x + lms

    def _make_resblocks(self,channels,block_num,isd):
        blocks=[]
        for i in range(block_num):
            blocks.append(("resblock_{}".format(i),Residual_Block(channels,isd)))
        return nn.Sequential(OrderedDict(blocks))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    from torchsummary import summary
    summary(SSCANet(channels=32,block_num=4,isd=True),input_size=[(1,64,64),(8,64,64)],device='cpu')

    print(get_parameter_number(SSCANet(32,4,True)))
